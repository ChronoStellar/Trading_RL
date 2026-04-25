//
//  SimulationViewModel.swift
//  PaperTrader
//
//  Runs the policy bar-by-bar over a user-selected historical window and
//  produces an equity curve vs. buy-and-hold for animated playback.
//

import Foundation
import CoreML
import SwiftYFinance
internal import Combine

struct SimStep {
    let date: Date
    let price: Double
    let allocation: Float
    let equity: Double      // agent portfolio value (starts at 1.0)
    let benchmark: Double   // buy-and-hold SPY-equivalent (starts at 1.0)
}

struct SimReport {
    let agentReturn: Double
    let benchmarkReturn: Double
    let sharpe: Double
    let maxDrawdown: Double
}

@MainActor
final class SimulationViewModel: ObservableObject {

    // MARK: Tunables (mirror the Gym env defaults)
    static let transactionCost: Double = 0.0005
    static let slippage: Double        = 0.0005

    // MARK: Published state
    @Published var ticker: String     = "SPY"
    @Published var searchText: String = "SPY"
    @Published var startDate: Date
    @Published var endDate: Date

    @Published var steps: [SimStep] = []
    @Published var visibleCount: Int = 0            // for animated reveal
    @Published var isLoading: Bool   = false
    @Published var isPlaying: Bool   = false
    @Published var errorMessage: String?

    var isComplete: Bool { !steps.isEmpty && visibleCount >= steps.count }

    var report: SimReport? {
        guard let last = steps.last, let first = steps.first else { return nil }
        let agentReturn     = last.equity / first.equity - 1
        let benchmarkReturn = last.benchmark / first.benchmark - 1
        let equities = steps.map(\.equity)
        let rets = zip(equities.dropFirst(), equities).map { (c, p) in (c - p) / p }
        let mean = rets.isEmpty ? 0 : rets.reduce(0, +) / Double(rets.count)
        let variance = rets.map { pow($0 - mean, 2) }.reduce(0, +) / Double(max(rets.count - 1, 1))
        let std = sqrt(variance)
        let sharpe = std > 0 ? (mean / std) * sqrt(252) : 0
        var peak = equities[0]; var maxDD = 0.0
        for e in equities { peak = max(peak, e); maxDD = min(maxDD, (e - peak) / peak) }
        return SimReport(agentReturn: agentReturn,
                         benchmarkReturn: benchmarkReturn,
                         sharpe: sharpe,
                         maxDrawdown: maxDD)
    }

    // MARK: Private
    private let model:  TradingActor
    private let engine: FeatureEngine
    private var playTask: Task<Void, Never>?

    // MARK: Init
    init() {
        guard let m = try? TradingActor() else {
            fatalError("Could not load TradingActor.mlpackage")
        }
        model  = m
        engine = FeatureEngine(scaler: loadScaler())
        let now = Date()
        self.endDate   = now
        self.startDate = Calendar.current.date(byAdding: .month, value: -6, to: now)!
    }

    // MARK: Public actions

    func selectTicker(_ t: String) {
        searchText = t
        ticker = t.uppercased()
    }

    func search() {
        let t = searchText.trimmingCharacters(in: .whitespaces).uppercased()
        guard !t.isEmpty else { return }
        ticker = t
    }

    /// Fetches bars and walks the policy forward across the range.
    func simulate() {
        playTask?.cancel()
        Task { await runSimulation() }
    }

    /// Plays the already-computed steps with animation.
    func play() {
        guard !steps.isEmpty else { return }
        playTask?.cancel()
        isPlaying = true
        visibleCount = 1
        playTask = Task { @MainActor in
            let total = steps.count
            // ~3 seconds total playback
            let stepDelay = UInt64(max(8_000_000, 3_000_000_000 / UInt64(total)))
            while visibleCount < total {
                try? await Task.sleep(nanoseconds: stepDelay)
                if Task.isCancelled { break }
                visibleCount = min(visibleCount + 1, total)
            }
            isPlaying = false
        }
    }

    private func runSimulation() async {
        isLoading = true
        errorMessage = nil
        steps = []
        visibleCount = 0

        // Fetch extra lead-in history for feature computation.
        let leadIn = Calendar.current.date(byAdding: .day, value: -120, to: startDate)!

        let history: [OHLCVBar]
        do {
            history = try await fetchBars(ticker: ticker, start: leadIn, end: endDate)
        } catch {
            errorMessage = "Fetch failed: \(error.localizedDescription)"
            isLoading = false
            return
        }

        let simBars = history.filter { $0.date >= startDate && $0.date <= endDate }
        guard simBars.count >= 2 else {
            errorMessage = "Not enough bars in selected range."
            isLoading = false
            return
        }

        guard let firstIdx = history.firstIndex(where: { $0.date >= startDate }),
              firstIdx >= FeatureEngine.minBars else {
            errorMessage = "Need \(FeatureEngine.minBars)+ trading days before the start date."
            isLoading = false
            return
        }

        var position: Double   = 0
        var entryPrice: Double = 0
        var equity: Double     = 1.0
        var bench: Double      = 1.0
        var computed: [SimStep] = []

        for i in firstIdx..<history.count {
            let window = Array(history[0...i])
            guard let obs = engine.observation(bars: window, position: position, entryPrice: entryPrice) else { continue }

            let alloc: Float
            do {
                let input = try mlInput(from: obs)
                let out   = try await model.prediction(input: input)
                alloc = out.allocation[0].floatValue
            } catch {
                errorMessage = "Inference error: \(error.localizedDescription)"
                isLoading = false
                return
            }

            let bar = history[i]
            // Apply the *previous* position to today's return, then rebalance to `alloc`.
            if i > firstIdx {
                let prev = history[i - 1].close
                let priceRet = (bar.close - prev) / prev
                equity *= (1.0 + position * priceRet)
                bench  *= (1.0 + priceRet)
            }

            // Transaction cost on rebalance
            let newPos = Double(alloc)
            let turnover = abs(newPos - position)
            equity *= (1.0 - (Self.transactionCost + Self.slippage) * turnover)

            if position == 0 && newPos > 0 { entryPrice = bar.close }
            else if newPos == 0            { entryPrice = 0 }
            position = newPos

            computed.append(SimStep(date: bar.date, price: bar.close,
                                    allocation: alloc, equity: equity, benchmark: bench))
        }

        steps = computed
        isLoading = false
        play()
    }

    // MARK: Helpers

    private func mlInput(from obs: [Float]) throws -> TradingActorInput {
        let array = try MLMultiArray(shape: [1, NSNumber(value: obs.count)], dataType: .float32)
        for (i, v) in obs.enumerated() { array[i] = NSNumber(value: v) }
        return TradingActorInput(observation: array)
    }

    private func fetchBars(ticker: String, start: Date, end: Date) async throws -> [OHLCVBar] {
        try await withCheckedThrowingContinuation { continuation in
            SwiftYFinance.chartDataBy(
                identifier: ticker, start: start, end: end, interval: .oneday
            ) { (data: [StockChartData]?, error: Error?) in
                if let error { continuation.resume(throwing: error); return }
                guard let data, !data.isEmpty else {
                    continuation.resume(throwing: URLError(.badServerResponse)); return
                }
                let bars: [OHLCVBar] = data.compactMap { bar in
                    guard let date  = bar.date,
                          let open  = bar.open,  let high = bar.high,
                          let low   = bar.low,   let close = bar.close,
                          let vol   = bar.volume else { return nil }
                    return OHLCVBar(date: date,
                                    open: Double(open), high: Double(high),
                                    low: Double(low),   close: Double(close),
                                    volume: Double(vol))
                }
                continuation.resume(returning: bars)
            }
        }
    }
}
