//
//  SimulationViewModel.swift
//  PaperTrader
//
//  Interactive simulation: pre-computes bars + the model's suggested allocations,
//  then plays the chart back bar-by-bar. At every "critical decision point" (model
//  allocation crosses an action threshold) the playback pauses and waits for the
//  user to Buy / Sell / Hold. The *user's* actions drive the equity curve.
//

import Foundation
import CoreML
import SwiftYFinance
internal import Combine

struct SimStep {
    let date: Date
    let price: Double
    let modelAlloc: Float   // model's suggested allocation at this bar
}

enum ActionKind { case buy, sell }

struct ActionPoint: Identifiable {
    let id = UUID()
    let kind: ActionKind        // direction the model suggests
    let date: Date
    let price: Double
    let modelAlloc: Float       // model's suggestion at the decision bar
    var index: Int              // index into `steps`
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
    let modelReturn: Double      // for comparison: what the agent would have earned alone
    let userActions: Int
    let modelActions: Int
}

@MainActor
final class SimulationViewModel: ObservableObject {

    // MARK: Tunables
    static let transactionCost: Double = 0.0005
    static let slippage:        Double = 0.0005
    static let actionThreshold: Float  = 0.5
    /// Per-bar playback delay during animation (ms).
    static let playbackTickMs:  UInt64 = 50

    // MARK: Configuration
    @Published var ticker: String     = "SPY"
    @Published var searchText: String = "SPY"
    @Published var startDate: Date
    @Published var endDate:   Date

    // MARK: Pre-computed series
    @Published var steps:   [SimStep]    = []
    @Published var actions: [ActionPoint] = []   // model's decision points

    // MARK: Live playback state
    @Published var revealedIndex: Int       = 0          // last bar shown on chart
    @Published var pendingDecision: ActionPoint?         // non-nil → playback paused
    @Published var userActions: [UserAction] = []
    @Published var userEquity:  [Double]     = [1.0]     // equity series, indexed alongside revealedIndex
    @Published var benchEquity: [Double]     = [1.0]
    @Published var modelEquity: [Double]     = [1.0]     // ghost: what model alone would do
    @Published var userPosition: Double      = 0
    @Published var isPlaying: Bool           = false
    @Published var isLoading: Bool           = false
    @Published var isComplete: Bool          = false
    @Published var errorMessage: String?

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

    // MARK: Private
    private let model:  TradingActor
    private let engine: FeatureEngine
    private var playTask: Task<Void, Never>?
    private var modelPosition: Double = 0     // for ghost equity series

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

    // MARK: Configuration actions

    func selectTicker(_ t: String) {
        searchText = t
        ticker = t.uppercased()
    }

    func search() {
        let t = searchText.trimmingCharacters(in: .whitespaces).uppercased()
        guard !t.isEmpty else { return }
        ticker = t
    }

    // MARK: Simulation lifecycle

    /// Resets state, fetches bars, precomputes the agent's decisions, then starts playback.
    func startSimulation() async {
        cancelPlayback()
        resetState()
        isLoading = true

        let leadIn = Calendar.current.date(byAdding: .day, value: -120, to: startDate)!
        let history: [OHLCVBar]
        do {
            history = try await fetchBars(ticker: ticker, start: leadIn, end: endDate)
        } catch {
            errorMessage = "Fetch failed: \(error.localizedDescription)"
            isLoading = false
            return
        }

        guard let firstIdx = history.firstIndex(where: { $0.date >= startDate }),
              firstIdx >= FeatureEngine.minBars else {
            errorMessage = "Need \(FeatureEngine.minBars)+ trading days before the start date."
            isLoading = false
            return
        }

        var ghostPosition: Double = 0
        var ghostEntry:    Double = 0
        var prevAlloc:     Float  = 0
        var computed:      [SimStep]    = []
        var detected:      [ActionPoint] = []

        for i in firstIdx..<history.count {
            let window = Array(history[0...i])
            guard let obs = engine.observation(bars: window, position: ghostPosition, entryPrice: ghostEntry) else { continue }

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
            let stepIndex = computed.count

            // Detect threshold crossings as action points.
            if !computed.isEmpty {
                if prevAlloc < Self.actionThreshold && alloc >= Self.actionThreshold {
                    detected.append(ActionPoint(kind: .buy,  date: bar.date, price: bar.close,
                                                modelAlloc: alloc, index: stepIndex))
                } else if prevAlloc >= Self.actionThreshold && alloc < Self.actionThreshold {
                    detected.append(ActionPoint(kind: .sell, date: bar.date, price: bar.close,
                                                modelAlloc: alloc, index: stepIndex))
                }
            }
            prevAlloc = alloc

            // Update ghost position bookkeeping.
            let newPos = Double(alloc)
            if ghostPosition == 0 && newPos > 0 { ghostEntry = bar.close }
            else if newPos == 0                 { ghostEntry = 0 }
            ghostPosition = newPos

            computed.append(SimStep(date: bar.date, price: bar.close, modelAlloc: alloc))
        }

        steps   = computed
        actions = detected
        isLoading = false
        guard !steps.isEmpty else {
            errorMessage = "No bars in simulation range."
            return
        }
        play()
    }

    /// User's response to a pending decision point. Resumes playback.
    func decide(_ choice: UserChoice) {
        guard let decision = pendingDecision else { return }
        let prev = userPosition
        let target = Double(decision.modelAlloc)
        let newPos: Double = {
            switch choice {
            // Accept the model's recommendation in the direction it suggested.
            case .buy:  return max(prev, target)
            case .sell: return min(prev, target)
            case .hold: return prev
            }
        }()

        // Record the action.
        userActions.append(UserAction(
            choice: choice,
            date: decision.date,
            price: decision.price,
            positionBefore: prev,
            positionAfter: newPos,
            stepIndex: decision.index
        ))

        // Apply transaction cost on the current (revealed) equity for any rebalance.
        let turnover = abs(newPos - prev)
        if turnover > 0, var last = userEquity.last {
            last *= (1.0 - (Self.transactionCost + Self.slippage) * turnover)
            userEquity[userEquity.count - 1] = last
        }

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
        guard revealedIndex < steps.count - 1 else {
            isComplete = revealedIndex == steps.count - 1 && !steps.isEmpty
            isPlaying = false
            return
        }
        isPlaying = true
        playTask = Task { @MainActor in
            while revealedIndex < steps.count - 1 {
                if Task.isCancelled { break }
                try? await Task.sleep(nanoseconds: Self.playbackTickMs * 1_000_000)
                if Task.isCancelled { break }
                advanceOneBar()
                if pendingDecision != nil {
                    isPlaying = false
                    return
                }
            }
            isComplete = !steps.isEmpty
            isPlaying = false
        }
    }

    private func advanceOneBar() {
        let next = revealedIndex + 1
        guard next < steps.count else { return }
        let prev = steps[revealedIndex]
        let curr = steps[next]
        let priceRet = (curr.price - prev.price) / prev.price

        // User equity uses user's current position.
        let lastUser  = userEquity.last  ?? 1.0
        let lastBench = benchEquity.last ?? 1.0
        let lastModel = modelEquity.last ?? 1.0
        userEquity.append(lastUser  * (1.0 + userPosition          * priceRet))
        benchEquity.append(lastBench * (1.0 + priceRet))
        modelEquity.append(lastModel * (1.0 + Double(prev.modelAlloc) * priceRet))

        revealedIndex = next

        // Pause for user input on the *bar* matching an action point.
        if let action = actions.first(where: { $0.index == next }) {
            pendingDecision = action
        }
    }

    private func resetState() {
        revealedIndex   = 0
        pendingDecision = nil
        userActions     = []
        userEquity      = [1.0]
        benchEquity     = [1.0]
        modelEquity     = [1.0]
        userPosition    = 0
        modelPosition   = 0
        steps           = []
        actions         = []
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
