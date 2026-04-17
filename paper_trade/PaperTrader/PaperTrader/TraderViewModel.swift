//
//  TraderViewModel.swift
//  PaperTrader
//
//  Fetches recent SPY OHLCV data via SwiftYFinance, computes the observation
//  vector, runs the CoreML actor, and exposes results to the SwiftUI view.
//

import Foundation
import CoreML
import SwiftYFinance
internal import Combine

@MainActor
final class TraderViewModel: ObservableObject {

    // MARK: - Published state

    @Published var allocation: Float    = 0        // model output  [0, 1]
    @Published var currentPrice: Double = 0
    @Published var position: Double     = 0        // current allocation applied
    @Published var unrealisedPnL: Double = 0       // (price / entryPrice) - 1
    @Published var isLoading: Bool      = false
    @Published var errorMessage: String?
    @Published var lastUpdated: Date?

    // MARK: - Private

    private let ticker      = "SPY"
    private var entryPrice: Double = 0
    private var bars: [OHLCVBar]   = []

    private let model:  TradingActor
    private let engine: FeatureEngine

    // MARK: - Init

    init() {
        guard let m = try? TradingActor() else {
            fatalError("Could not load TradingActor.mlpackage — add it to the Xcode target.")
        }
        model  = m
        engine = FeatureEngine(scaler: loadScaler())
    }

    // MARK: - Public actions

    func refresh() {
        Task { await fetchAndInfer() }
    }

    /// Record the position recommended by the model and update entry price.
    func applyAllocation() {
        let prev    = position
        position    = Double(allocation)
        if prev == 0 && position > 0 {
            entryPrice = currentPrice   // opening a new position
        } else if position == 0 {
            entryPrice = 0
        }
        unrealisedPnL = entryPrice > 0 ? (currentPrice / entryPrice) - 1 : 0
    }

    // MARK: - Data fetch + inference

    private func fetchAndInfer() async {
        isLoading     = true
        errorMessage  = nil

        // Fetch 40 calendar days so we always have ≥ 21 trading days
        let end   = Date()
        let start = Calendar.current.date(byAdding: .day, value: -40, to: end)!

        do {
            bars         = try await fetchBars(ticker: ticker, start: start, end: end)
            currentPrice = bars.last?.close ?? 0
        } catch {
            errorMessage = "Fetch failed: \(error.localizedDescription)"
            isLoading    = false
            return
        }

        // Build observation
        guard let obs = engine.observation(
            bars: bars, position: position, entryPrice: entryPrice
        ) else {
            errorMessage = "Not enough data — need \(FeatureEngine.minBars)+ trading days."
            isLoading    = false
            return
        }

        // CoreML inference
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

    // MARK: - Helpers

    private func mlInput(from obs: [Float]) throws -> TradingActorInput {
        let array = try MLMultiArray(shape: [1, NSNumber(value: obs.count)], dataType: .float32)
        for (i, v) in obs.enumerated() { array[i] = NSNumber(value: v) }
        return TradingActorInput(observation: array)
    }

    /// Async wrapper around SwiftYFinance's callback-based chart API.
    /// `chartDataBy` returns `[StockChartData]?` — a flat array where each element
    /// is one OHLCV bar with Float? price fields and an Int? volume field.
    private func fetchBars(ticker: String, start: Date, end: Date) async throws -> [OHLCVBar] {
        try await withCheckedThrowingContinuation { continuation in
            SwiftYFinance.chartDataBy(
                identifier: ticker,
                start: start,
                end: end,
                interval: .oneday
            ) { (data: [StockChartData]?, error: Error?) in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                guard let data, !data.isEmpty else {
                    continuation.resume(throwing: URLError(.badServerResponse))
                    return
                }

                let bars: [OHLCVBar] = data.compactMap { bar in
                    guard
                        let date   = bar.date,
                        let open   = bar.open,
                        let high   = bar.high,
                        let low    = bar.low,
                        let close  = bar.close,
                        let volume = bar.volume
                    else { return nil }
                    return OHLCVBar(
                        date:   date,
                        open:   Double(open),
                        high:   Double(high),
                        low:    Double(low),
                        close:  Double(close),
                        volume: Double(volume)
                    )
                }

                continuation.resume(returning: bars)
            }
        }
    }
}
