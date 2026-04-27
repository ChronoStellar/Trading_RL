//
//  SimulationViewModel.swift
//  PaperTrader
//
//  Interactive simulation with **live inference**: the model is run once per
//  playback tick, conditioned on the user's *actual* position. Whenever the
//  model's allocation crosses the action threshold, playback pauses and asks
//  the user to Buy / Sell / Hold. A parallel "ghost" pass (model running
//  itself) tracks what the agent alone would have done, for comparison.
//

import Foundation
import CoreML
import SwiftYFinance
internal import Combine

struct SimStep {
    let date: Date
    let price: Double
    let modelAlloc: Float   // model's recommendation given the user's actual position
}

enum ActionKind { case buy, sell }

struct ActionPoint: Identifiable {
    let id = UUID()
    let kind: ActionKind
    let date: Date
    let price: Double
    let modelAlloc: Float
    var index: Int
}

enum UserChoice { case buy, sell, hold }

struct UserAction: Identifiable {
    let id = UUID()
    let choice: UserChoice
    let date: Date
    let price: Double
    let positionBefore: Double
    let positionAfter: Double
    let stepIndex: Int
}

struct SimReport {
    let userReturn: Double
    let benchmarkReturn: Double
    let modelReturn: Double
    let userActions: Int
    let modelActions: Int
}

@MainActor
final class SimulationViewModel: ObservableObject {

    // MARK: Tunables
    static let transactionCost: Double = 0.0005
    static let slippage:        Double = 0.0005
    static let actionThreshold: Float  = 0.5
    /// Per-bar playback delay during animation (ms). Live inference happens
    /// inside the loop, so the wall-clock cadence is `tick + inference`.
    static let playbackTickMs:  UInt64 = 150

    // MARK: Configuration
    @Published var ticker: String     = "SPY"
    @Published var searchText: String = "SPY"
    @Published var startDate: Date
    @Published var endDate:   Date

    // MARK: Live state (grown bar-by-bar during playback)
    @Published var steps:        [SimStep]    = []
    @Published var actions:      [ActionPoint] = []
    @Published var pendingDecision: ActionPoint?
    @Published var userActions:  [UserAction] = []
    @Published var userEquity:   [Double] = [1.0]
    @Published var benchEquity:  [Double] = [1.0]
    @Published var modelEquity:  [Double] = [1.0]
    @Published var userPosition: Double  = 0
    @Published var isPlaying:    Bool    = false
    @Published var isLoading:    Bool    = false
    @Published var isComplete:   Bool    = false
    @Published var errorMessage: String?

    /// `revealedIndex` is `steps.count - 1` once at least one step exists,
    /// else `-1`. The view uses this for "Bar N / total".
    var revealedIndex: Int { max(steps.count - 1, 0) }
    var totalBars:     Int { history.count - simStartIdx }

    var report: SimReport? {
        guard isComplete, let last = userEquity.last, let first = userEquity.first else { return nil }
        return SimReport(
            userReturn:      last / first - 1,
            benchmarkReturn: (benchEquity.last ?? 1) / (benchEquity.first ?? 1) - 1,
            modelReturn:     (modelEquity.last ?? 1) / (modelEquity.first ?? 1) - 1,
            userActions:     userActions.count,
            modelActions:    actions.count
        )
    }

    // MARK: Private — pre-fetch
    private var history:     [OHLCVBar] = []
    private var simStartIdx: Int = 0   // index in `history` of the first bar in [startDate, endDate]
    private var cursor:      Int = 0   // index in `history` of the next bar to reveal

    // MARK: Private — running state
    private var entryPrice:     Double = 0   // user position entry (for obs's equity_return proxy)
    private var ghostPosition:  Double = 0
    private var ghostEntry:     Double = 0
    private var prevModelAlloc: Float  = 0

    // MARK: Private — engine
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

    // MARK: Configuration

    func selectTicker(_ t: String) {
        searchText = t
        ticker = t.uppercased()
    }

    func search() {
        let t = searchText.trimmingCharacters(in: .whitespaces).uppercased()
        guard !t.isEmpty else { return }
        ticker = t
    }

    // MARK: Lifecycle

    /// Fetches bars (with feature lead-in) then begins live playback.
    func startSimulation() async {
        cancelPlayback()
        resetState()
        isLoading = true

        let leadIn = Calendar.current.date(byAdding: .day, value: -120, to: startDate)!
        let fetched: [OHLCVBar]
        do {
            fetched = try await fetchBars(ticker: ticker, start: leadIn, end: endDate)
        } catch {
            errorMessage = "Fetch failed: \(error.localizedDescription)"
            isLoading = false
            return
        }

        guard let firstIdx = fetched.firstIndex(where: { $0.date >= startDate }),
              firstIdx >= FeatureEngine.minBars else {
            errorMessage = "Need \(FeatureEngine.minBars)+ trading days before the start date."
            isLoading = false
            return
        }

        history      = fetched
        simStartIdx  = firstIdx
        cursor       = firstIdx
        isLoading    = false
        play()
    }

    func decide(_ choice: UserChoice) {
        guard let decision = pendingDecision else { return }
        let prev   = userPosition
        let target = Double(decision.modelAlloc)
        let newPos: Double = {
            switch choice {
            case .buy:  return max(prev, target)
            case .sell: return min(prev, target)
            case .hold: return prev
            }
        }()

        userActions.append(UserAction(
            choice: choice,
            date: decision.date,
            price: decision.price,
            positionBefore: prev,
            positionAfter: newPos,
            stepIndex: decision.index
        ))

        // Apply transaction cost on the current equity for any rebalance.
        let turnover = abs(newPos - prev)
        if turnover > 0, var last = userEquity.last {
            last *= (1.0 - (Self.transactionCost + Self.slippage) * turnover)
            userEquity[userEquity.count - 1] = last
        }

        if prev == 0 && newPos > 0 { entryPrice = decision.price }
        else if newPos == 0        { entryPrice = 0 }
        userPosition = newPos

        pendingDecision = nil
        play()
    }

    func cancelPlayback() {
        playTask?.cancel()
        playTask = nil
        isPlaying = false
    }

    // MARK: Playback engine

    private func play() {
        guard pendingDecision == nil else { return }
        guard cursor < history.count else {
            isComplete = !steps.isEmpty
            isPlaying = false
            return
        }
        isPlaying = true
        playTask = Task { @MainActor in
            while cursor < history.count {
                if Task.isCancelled { break }
                try? await Task.sleep(nanoseconds: Self.playbackTickMs * 1_000_000)
                if Task.isCancelled { break }
                await advanceOneBar()
                if pendingDecision != nil {
                    isPlaying = false
                    return
                }
            }
            isComplete = !steps.isEmpty
            isPlaying = false
        }
    }

    /// Reveals the next bar: live inference (user-conditioned), parallel ghost
    /// inference, equity update, threshold check.
    private func advanceOneBar() async {
        let i = cursor
        guard i < history.count else { return }

        let window = Array(history[0...i])

        // Live inference using the user's *actual* position.
        guard let userObs = engine.observation(bars: window, position: userPosition, entryPrice: entryPrice) else {
            // Insufficient history at this bar — skip forward without halting.
            cursor += 1
            return
        }
        guard let userInput = try? mlInput(from: userObs),
              let userOut   = try? await model.prediction(input: userInput) else {
            errorMessage = "Inference error at bar \(i)."
            cursor = history.count
            return
        }
        let userAlloc = userOut.allocation[0].floatValue

        // Ghost inference (model running itself) for the comparison curve.
        var ghostAlloc: Float = 0
        if let ghostObs = engine.observation(bars: window, position: ghostPosition, entryPrice: ghostEntry),
           let ghostInput = try? mlInput(from: ghostObs),
           let ghostOut   = try? await model.prediction(input: ghostInput) {
            ghostAlloc = ghostOut.allocation[0].floatValue
        }

        let bar = history[i]
        let isFirstStep = steps.isEmpty

        // Compute per-bar return and update equities (skip on first step — no prior bar).
        if !isFirstStep {
            let prev = history[i - 1].close
            let priceRet = (bar.close - prev) / prev

            let lastUser  = userEquity.last  ?? 1.0
            let lastBench = benchEquity.last ?? 1.0
            let lastModel = modelEquity.last ?? 1.0
            userEquity.append(lastUser  * (1.0 + userPosition         * priceRet))
            benchEquity.append(lastBench * (1.0 + priceRet))
            modelEquity.append(lastModel * (1.0 + ghostPosition       * priceRet))
        }

        // Update ghost bookkeeping for the next bar.
        let newGhost = Double(ghostAlloc)
        if ghostPosition == 0 && newGhost > 0 { ghostEntry = bar.close }
        else if newGhost == 0                 { ghostEntry = 0 }
        ghostPosition = newGhost

        // Append the step *before* deciding on a pause so the chart shows it.
        let stepIndex = steps.count
        steps.append(SimStep(date: bar.date, price: bar.close, modelAlloc: userAlloc))

        // Threshold-cross detection on the user-conditioned signal.
        if !isFirstStep {
            if prevModelAlloc < Self.actionThreshold && userAlloc >= Self.actionThreshold {
                let pt = ActionPoint(kind: .buy, date: bar.date, price: bar.close,
                                     modelAlloc: userAlloc, index: stepIndex)
                actions.append(pt)
                pendingDecision = pt
            } else if prevModelAlloc >= Self.actionThreshold && userAlloc < Self.actionThreshold {
                let pt = ActionPoint(kind: .sell, date: bar.date, price: bar.close,
                                     modelAlloc: userAlloc, index: stepIndex)
                actions.append(pt)
                pendingDecision = pt
            }
        }
        prevModelAlloc = userAlloc

        cursor += 1
    }

    private func resetState() {
        history         = []
        simStartIdx     = 0
        cursor          = 0
        steps           = []
        actions         = []
        userActions     = []
        userEquity      = [1.0]
        benchEquity     = [1.0]
        modelEquity     = [1.0]
        userPosition    = 0
        entryPrice      = 0
        ghostPosition   = 0
        ghostEntry      = 0
        prevModelAlloc  = 0
        pendingDecision = nil
        isComplete      = false
        errorMessage    = nil
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
