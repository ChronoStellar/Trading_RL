//
//  TraderViewModel.swift
//  PaperTrader
//

import Foundation
import CoreML
import SwiftYFinance
internal import Combine

// MARK: - Date range preset

enum DateRange: String, CaseIterable, Identifiable {
    case oneMonth  = "1M"
    case threeMonths = "3M"
    case sixMonths = "6M"
    case oneYear   = "1Y"

    var id: String { rawValue }

    var calendarDays: Int {
        switch self {
        case .oneMonth:    return 40    // 40 calendar ≈ 21+ trading days (min for features)
        case .threeMonths: return 95
        case .sixMonths:   return 185
        case .oneYear:     return 370
        }
    }
}

// MARK: - ViewModel

@MainActor
final class TraderViewModel: ObservableObject {

    // MARK: Published state

    @Published var ticker: String      = "SPY"
    @Published var searchText: String  = "SPY"
    @Published var dateRange: DateRange = .threeMonths

    @Published var bars: [OHLCVBar]    = []
    @Published var allocation: Float   = 0
    @Published var currentPrice: Double = 0
    @Published var position: Double    = 0
    @Published var unrealisedPnL: Double = 0
    @Published var isLoading: Bool     = false
    @Published var errorMessage: String?
    @Published var lastUpdated: Date?

    // MARK: Private

    private var entryPrice: Double = 0
    private let model:  TradingActor
    private let engine: FeatureEngine

    // MARK: Init

    init() {
        guard let m = try? TradingActor() else {
            fatalError("Could not load TradingActor.mlpackage")
        }
        model  = m
        engine = FeatureEngine(scaler: loadScaler())
    }

    // MARK: Public actions

    func search() {
        let t = searchText.trimmingCharacters(in: .whitespaces).uppercased()
        guard !t.isEmpty else { return }
        ticker = t
        Task { await fetchAndInfer() }
    }

    func selectDateRange(_ range: DateRange) {
        dateRange = range
        Task { await fetchAndInfer() }
    }

    func refresh() {
        Task { await fetchAndInfer() }
    }

    func applyAllocation() {
        let prev = position
        position = Double(allocation)
        if prev == 0 && position > 0 {
            entryPrice = currentPrice
        } else if position == 0 {
            entryPrice = 0
        }
        unrealisedPnL = entryPrice > 0 ? (currentPrice / entryPrice) - 1 : 0
    }

    // MARK: Fetch + infer

    private func fetchAndInfer() async {
        isLoading    = true
        errorMessage = nil

        let end   = Date()
        let start = Calendar.current.date(byAdding: .day, value: -dateRange.calendarDays, to: end)!

        do {
            bars         = try await fetchBars(ticker: ticker, start: start, end: end)
            currentPrice = bars.last?.close ?? 0
        } catch {
            errorMessage = "Fetch failed: \(error.localizedDescription)"
            isLoading    = false
            return
        }

        // Need at least minBars to compute all rolling features
        guard let obs = engine.observation(
            bars: bars, position: position, entryPrice: entryPrice
        ) else {
            errorMessage = "Need \(FeatureEngine.minBars)+ trading days — try a longer range."
            isLoading    = false
            return
        }

        do {
            let input  = try mlInput(from: obs)
            let output = try await model.prediction(input: input)
            allocation = output.allocation[0].floatValue
        } catch {
            errorMessage = "Inference error: \(error.localizedDescription)"
            isLoading    = false
            return
        }

        unrealisedPnL = entryPrice > 0 ? (currentPrice / entryPrice) - 1 : 0
        lastUpdated   = Date()
        isLoading     = false
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
                identifier: ticker,
                start: start,
                end: end,
                interval: .oneday
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
