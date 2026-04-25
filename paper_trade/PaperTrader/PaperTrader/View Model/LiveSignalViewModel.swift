//
//  LiveSignalViewModel.swift
//  PaperTrader
//

import Foundation
import CoreML
import SwiftYFinance
internal import Combine

enum DisplayWindow: String, CaseIterable, Identifiable {
    case oneDay      = "1D"
    case fiveDay     = "5D"
    case oneMonth    = "1M"
    case threeMonths = "3M"
    case sixMonths   = "6M"
    case oneYear     = "1Y"

    var id: String { rawValue }

    /// How many of the most recent *trading* bars to display on the chart.
    var displayBars: Int {
        switch self {
        case .oneDay:      return 1
        case .fiveDay:     return 5
        case .oneMonth:    return 22
        case .threeMonths: return 66
        case .sixMonths:   return 130
        case .oneYear:     return 252
        }
    }
}

@MainActor
final class LiveSignalViewModel: ObservableObject {

    // MARK: Published state
    @Published var ticker: String       = "SPY"
    @Published var searchText: String   = "SPY"
    @Published var window: DisplayWindow = .sixMonths

    @Published var bars: [OHLCVBar]     = []   // full fetched history (for feature computation)
    @Published var allocation: Float    = 0
    @Published var currentPrice: Double = 0
    @Published var isLoading: Bool      = false
    @Published var errorMessage: String?
    @Published var lastUpdated: Date?

    /// Bars visible in the chart — suffix of `bars` per selected window.
    var displayedBars: [OHLCVBar] {
        Array(bars.suffix(window.displayBars))
    }

    // MARK: Private
    private let model:  TradingActor
    private let engine: FeatureEngine
    /// Calendar-day fetch window — large enough for feature computation (≥ 61 trading days) *and* up to 1Y display.
    private let fetchCalendarDays = 400

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

    func selectTicker(_ t: String) {
        searchText = t
        search()
    }

    func selectWindow(_ w: DisplayWindow) {
        window = w  // no refetch — the same bars power the chart slice
    }

    func refresh() {
        Task { await fetchAndInfer() }
    }

    // MARK: Fetch + infer

    private func fetchAndInfer() async {
        isLoading    = true
        errorMessage = nil

        let end   = Date()
        let start = Calendar.current.date(byAdding: .day, value: -fetchCalendarDays, to: end)!

        do {
            bars         = try await fetchBars(ticker: ticker, start: start, end: end)
            currentPrice = bars.last?.close ?? 0
        } catch {
            errorMessage = "Fetch failed: \(error.localizedDescription)"
            isLoading    = false
            return
        }

        guard let obs = engine.observation(bars: bars, position: 0, entryPrice: 0) else {
            errorMessage = "Need \(FeatureEngine.minBars)+ trading days of history."
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

        lastUpdated = Date()
        isLoading   = false
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
