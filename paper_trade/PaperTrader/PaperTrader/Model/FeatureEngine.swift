//
//  FeatureEngine.swift
//  PaperTrader
//
//  Replicates the feature computation from data/features.py so live OHLCV bars
//  produce an observation vector identical to what the model was trained on.
//
//  Observation layout (10-d):
//    [0] sma_ratio    [1] rsi_14      [2] macd_hist
//    [3] obv_ret      [4] adx         [5] drawdown_60d
//    [6] vol_regime   [7] ret_20d     [8] position     [9] equity_return
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

    /// Must match FEATURE_COLS order in data/features.py exactly.
    static let featureCols = [
        "sma_ratio", "rsi_14", "macd_hist",
        "obv_ret", "adx",
        "drawdown_60d", "vol_regime", "ret_20d",
    ]

    /// Minimum bars required: drawdown_60d and vol_regime need 60 daily returns = 61 prices.
    static let minBars = 61

    let scaler: [String: FeatureScaler]

    // MARK: Public

    /// Returns a normalised 10-d observation: 8 market features + position + equity_return.
    /// Returns nil when there are fewer than `minBars` bars available.
    func observation(bars: [OHLCVBar], position: Double, entryPrice: Double) -> [Float]? {
        guard bars.count >= Self.minBars else { return nil }

        let closes  = bars.map(\.close)
        let highs   = bars.map(\.high)
        let lows    = bars.map(\.low)
        let volumes = bars.map(\.volume)

        let raw: [String: Double] = [
            "sma_ratio"   : smaRatio(closes, window: 20),
            "rsi_14"      : rsi(closes, period: 14),
            "macd_hist"   : macdHist(closes),
            "obv_ret"     : obvRet(closes: closes, volumes: volumes),
            "adx"         : adx(highs: highs, lows: lows, closes: closes, period: 14),
            "drawdown_60d": drawdown60d(closes),
            "vol_regime"  : volRegime(closes),
            "ret_20d"     : ret20d(closes),
        ]

        var obs = Self.featureCols.map { col -> Float in
            let v = raw[col] ?? 0
            return Float(scaler[col]?.normalize(v) ?? v)
        }

        // Env-injected features (not normalised — same as the Gym env).
        // equity_return proxy: unrealised P&L of current position since entry.
        let equityReturn = entryPrice > 0 ? (closes.last! / entryPrice) - 1.0 : 0.0
        obs.append(Float(position))
        obs.append(Float(equityReturn))

        return obs  // length 10
    }

    // MARK: - Feature helpers (mirror data/features.py)

    // sma_ratio = close / SMA-20
    private func smaRatio(_ closes: [Double], window: Int) -> Double {
        let sma = closes.suffix(window).reduce(0, +) / Double(window)
        return closes.last! / sma
    }

    // Wilder's EWM RSI (alpha = 1/period, adjust=False) — matches ta.momentum.RSIIndicator
    private func rsi(_ closes: [Double], period: Int) -> Double {
        let alpha = 1.0 / Double(period)
        var avgGain = 0.0
        var avgLoss = 0.0
        for i in 1 ..< closes.count {
            let d = closes[i] - closes[i - 1]
            avgGain = alpha * max(d, 0)  + (1 - alpha) * avgGain
            avgLoss = alpha * max(-d, 0) + (1 - alpha) * avgLoss
        }
        guard avgLoss > 0 else { return 100 }
        return 100 - (100 / (1 + avgGain / avgLoss))
    }

    // MACD histogram / close — matches ta.trend.MACD(fast=12, slow=26, sign=9)
    // Uses pandas-style EWM: alpha = 2/(span+1), adjust=False.
    private func macdHist(_ closes: [Double]) -> Double {
        let ema12   = ewm(closes, alpha: 2.0 / 13)
        let ema26   = ewm(closes, alpha: 2.0 / 27)
        let macd    = zip(ema12, ema26).map { $0 - $1 }
        let signal  = ewm(macd,  alpha: 2.0 / 10)
        let hist    = zip(macd, signal).map { $0 - $1 }
        guard let last = hist.last, let close = closes.last, close != 0 else { return 0 }
        return last / close
    }

    // EWM with adjust=False: y[0]=x[0], y[t] = alpha*x[t] + (1-alpha)*y[t-1]
    private func ewm(_ values: [Double], alpha: Double) -> [Double] {
        guard !values.isEmpty else { return [] }
        var result = [Double]()
        result.reserveCapacity(values.count)
        var prev = values[0]
        result.append(prev)
        for v in values.dropFirst() {
            prev = alpha * v + (1 - alpha) * prev
            result.append(prev)
        }
        return result
    }

    // OBV 1-day % change — matches ta.volume.OnBalanceVolumeIndicator + pct_change(1)
    private func obvRet(closes: [Double], volumes: [Double]) -> Double {
        var obv = 0.0
        var prevObv = 0.0
        for i in 1 ..< closes.count {
            let d = closes[i] - closes[i - 1]
            prevObv = obv
            if d > 0      { obv += volumes[i] }
            else if d < 0 { obv -= volumes[i] }
        }
        guard prevObv != 0 else { return 0 }
        return obv / prevObv - 1
    }

    // ADX (14) — matches ta.trend.ADXIndicator using Wilder's EWM (alpha = 1/period, adjust=False)
    private func adx(highs: [Double], lows: [Double], closes: [Double], period: Int) -> Double {
        let n = closes.count
        guard n >= 2 else { return 0 }
        let alpha = 1.0 / Double(period)

        // Seed smoothed values with the first valid bar (index 1)
        let up1   = highs[1] - highs[0]
        let down1 = lows[0]  - lows[1]
        var smoothPlus  = (up1 > down1 && up1 > 0) ? up1 : 0.0
        var smoothMinus = (down1 > up1 && down1 > 0) ? down1 : 0.0
        var smoothTR    = max(highs[1] - lows[1],
                              abs(highs[1] - closes[0]),
                              abs(lows[1]  - closes[0]))

        // Seed ADX with DX at index 1
        var adxVal: Double = 0
        if smoothTR > 0 {
            let dp = 100 * smoothPlus  / smoothTR
            let dm = 100 * smoothMinus / smoothTR
            let s  = dp + dm
            adxVal = s > 0 ? 100 * abs(dp - dm) / s : 0
        }

        for i in 2 ..< n {
            let up   = highs[i] - highs[i - 1]
            let down = lows[i - 1] - lows[i]
            let plusDM  = (up > down && up > 0) ? up : 0.0
            let minusDM = (down > up && down > 0) ? down : 0.0
            let tr      = max(highs[i] - lows[i],
                              abs(highs[i] - closes[i - 1]),
                              abs(lows[i]  - closes[i - 1]))

            smoothTR    = alpha * tr      + (1 - alpha) * smoothTR
            smoothPlus  = alpha * plusDM  + (1 - alpha) * smoothPlus
            smoothMinus = alpha * minusDM + (1 - alpha) * smoothMinus

            if smoothTR > 0 {
                let dp  = 100 * smoothPlus  / smoothTR
                let dm  = 100 * smoothMinus / smoothTR
                let s   = dp + dm
                let dx  = s > 0 ? 100 * abs(dp - dm) / s : 0
                adxVal  = alpha * dx + (1 - alpha) * adxVal
            }
        }
        return adxVal
    }

    // drawdown_60d = close / max(close[-60:]) - 1  (≤ 0)
    private func drawdown60d(_ closes: [Double]) -> Double {
        let peak = closes.suffix(60).max() ?? closes.last!
        return (closes.last! / peak) - 1
    }

    // vol_regime = vol_20d / vol_60d  (> 1 = expanding)
    private func volRegime(_ closes: [Double]) -> Double {
        let rets  = zip(closes.dropFirst(), closes).map { $0 / $1 - 1 }
        let vol20 = stdDev(Array(rets.suffix(20)))
        let vol60 = stdDev(Array(rets.suffix(60)))
        return vol60 > 0 ? vol20 / vol60 : 1
    }

    // ret_20d = close / close[-21] - 1
    private func ret20d(_ closes: [Double]) -> Double {
        let n = closes.count
        guard n > 20 else { return 0 }
        return closes[n - 1] / closes[n - 21] - 1
    }

    private func stdDev(_ values: [Double]) -> Double {
        guard values.count > 1 else { return 0 }
        let mean     = values.reduce(0, +) / Double(values.count)
        let variance = values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(values.count - 1)
        return variance.squareRoot()
    }
}
