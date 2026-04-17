//
//  FeatureEngine.swift
//  PaperTrader
//
//  Replicates the feature computation from data/features.py so live OHLCV bars
//  produce an observation vector identical to what the model was trained on.
//

import Foundation

// MARK: - Data types

struct OHLCVBar {
    let date: Date
    let open: Double
    let high: Double
    let low: Double
    let close: Double
    let volume: Double
}

struct FeatureScaler: Decodable {
    let mean: Double
    let std: Double

    func normalize(_ value: Double) -> Double {
        (value - mean) / (std > 0 ? std : 1.0)
    }
}

// MARK: - Scaler loader

func loadScaler() -> [String: FeatureScaler] {
    guard
        let url  = Bundle.main.url(forResource: "scaler", withExtension: "json"),
        let data = try? Data(contentsOf: url),
        let dict = try? JSONDecoder().decode([String: FeatureScaler].self, from: data)
    else {
        return [:]
    }
    return dict
}

// MARK: - Feature engine

struct FeatureEngine {

    /// Must match the order in env/train_env.py FEATURE_COLS
    static let featureCols = ["ret_1d", "ret_5d", "rsi_14", "sma_ratio", "vol_20d", "vol_ratio"]

    /// Minimum bars required (SMA-20 / vol-20 need a 20-bar window + 5 for ret_5d)
    static let minBars = 21

    let scaler: [String: FeatureScaler]

    // MARK: Public

    /// Returns a normalised 8-d observation: 6 market features + position + unrealised PnL.
    /// Returns nil when there are fewer than `minBars` bars available.
    func observation(bars: [OHLCVBar], position: Double, entryPrice: Double) -> [Float]? {
        guard bars.count >= Self.minBars else { return nil }

        let closes  = bars.map(\.close)
        let volumes = bars.map(\.volume)

        let raw: [String: Double] = [
            "ret_1d"   : ret1d(closes),
            "ret_5d"   : ret5d(closes),
            "rsi_14"   : rsi(closes, period: 14),
            "sma_ratio": smaRatio(closes, window: 20),
            "vol_20d"  : vol20d(closes),
            "vol_ratio": volRatio(volumes),
        ]

        var obs = Self.featureCols.map { col -> Float in
            let v = raw[col] ?? 0
            return Float(scaler[col]?.normalize(v) ?? v)
        }

        // Append position state (not normalised — same as the Gym env)
        let unrealisedPnL = entryPrice > 0 ? (closes.last! / entryPrice) - 1.0 : 0.0
        obs.append(Float(position))
        obs.append(Float(unrealisedPnL))

        return obs   // length 8
    }

    // MARK: - Feature helpers (mirror data/features.py)

    private func ret1d(_ c: [Double]) -> Double {
        c[c.count - 1] / c[c.count - 2] - 1
    }

    private func ret5d(_ c: [Double]) -> Double {
        c[c.count - 1] / c[c.count - 6] - 1
    }

    /// Wilder's EWM RSI — matches features.py
    private func rsi(_ closes: [Double], period: Int) -> Double {
        let alpha = 1.0 / Double(period)
        var avgGain = 0.0
        var avgLoss = 0.0
        for i in 1 ..< closes.count {
            let d = closes[i] - closes[i - 1]
            avgGain = alpha * max(d, 0) + (1 - alpha) * avgGain
            avgLoss = alpha * max(-d, 0) + (1 - alpha) * avgLoss
        }
        guard avgLoss > 0 else { return 100 }
        return 100 - (100 / (1 + avgGain / avgLoss))
    }

    private func smaRatio(_ closes: [Double], window: Int) -> Double {
        let sma = closes.suffix(window).reduce(0, +) / Double(window)
        return closes.last! / sma
    }

    /// Rolling 20-day std of daily returns
    private func vol20d(_ closes: [Double]) -> Double {
        let rets = zip(closes.dropFirst(), closes).map { $0 / $1 - 1 }
        return stdDev(Array(rets.suffix(20)))
    }

    private func volRatio(_ volumes: [Double]) -> Double {
        let avg = volumes.suffix(20).reduce(0, +) / 20
        return avg > 0 ? (volumes.last! / avg) : 1
    }

    private func stdDev(_ values: [Double]) -> Double {
        guard values.count > 1 else { return 0 }
        let mean = values.reduce(0, +) / Double(values.count)
        let variance = values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(values.count - 1)
        return variance.squareRoot()
    }
}
