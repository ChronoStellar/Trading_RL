//
//  CandlestickChartView.swift
//  PaperTrader
//
//  Renders OHLCV bars using Swift Charts.
//  Requires iOS 16+ / Xcode 14+.
//

import SwiftUI
import Charts

struct CandlestickChartView: View {
    let bars: [OHLCVBar]

    // Thin out labels when there are many bars
    private var labelStride: Int {
        switch bars.count {
        case ..<30:  return 5
        case ..<90:  return 15
        default:     return 30
        }
    }

    var body: some View {
        Chart {
            ForEach(Array(bars.enumerated()), id: \.offset) { idx, bar in
                // Wick: high → low
                RuleMark(
                    x: .value("Date", bar.date),
                    yStart: .value("Low",  bar.low),
                    yEnd:   .value("High", bar.high)
                )
                .lineStyle(StrokeStyle(lineWidth: 1))
                .foregroundStyle(bar.close >= bar.open ? .green : .red)

                // Body: open ↔ close
                RectangleMark(
                    x:      .value("Date",  bar.date),
                    yStart: .value("Open",  min(bar.open, bar.close)),
                    yEnd:   .value("Close", max(bar.open, bar.close)),
                    width:  candleWidth
                )
                .foregroundStyle(bar.close >= bar.open ? .green : .red)
            }
        }
        .chartXAxis {
            AxisMarks(values: xAxisValues) { value in
                AxisGridLine()
                AxisValueLabel(format: .dateTime.month(.abbreviated).day())
            }
        }
        .chartYAxis {
            AxisMarks(position: .trailing) { value in
                AxisGridLine()
                AxisValueLabel(format: .currency(code: "USD").precision(.fractionLength(0)))
            }
        }
        .frame(height: 260)
    }

    // Pick evenly spaced dates for x-axis labels
    private var xAxisValues: [Date] {
        guard bars.count > 1 else { return [] }
        let step = max(1, bars.count / 5)
        return stride(from: 0, to: bars.count, by: step).map { bars[$0].date }
    }

    private var candleWidth: MarkDimension {
        switch bars.count {
        case ..<30:  return .fixed(8)
        case ..<90:  return .fixed(4)
        default:     return .fixed(2)
        }
    }
}
