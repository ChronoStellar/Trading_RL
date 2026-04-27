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
    var height: CGFloat? = nil  // nil = expands to fill available space

    // Per-candle horizontal slot. Bigger = more readable, requires scrolling.
    private let slotWidth: CGFloat = 14
    private let bodyWidth: CGFloat = 9
    private let yAxisGutter: CGFloat = 56

    var body: some View {
        GeometryReader { geo in
            let contentWidth = max(CGFloat(bars.count) * slotWidth, geo.size.width - yAxisGutter)

            HStack(spacing: 0) {
                scrollingChart(contentWidth: contentWidth)
                yAxis
                    .frame(width: yAxisGutter)
            }
        }
        .frame(minHeight: 180, maxHeight: height ?? .infinity)
    }

    @ViewBuilder
    private func scrollingChart(contentWidth: CGFloat) -> some View {
        if #available(iOS 17.0, *) {
            ScrollView(.horizontal, showsIndicators: false) {
                chart
                    .frame(width: contentWidth)
                    .padding(.trailing, 4)
            }
            .defaultScrollAnchor(.trailing)
        } else {
            ScrollView(.horizontal, showsIndicators: false) {
                chart
                    .frame(width: contentWidth)
                    .padding(.trailing, 4)
            }
        }
    }

    private var chart: some View {
        Chart {
            ForEach(Array(bars.enumerated()), id: \.offset) { _, bar in
                RuleMark(
                    x: .value("Date", bar.date),
                    yStart: .value("Low",  bar.low),
                    yEnd:   .value("High", bar.high)
                )
                .lineStyle(StrokeStyle(lineWidth: 1.2))
                .foregroundStyle(bar.close >= bar.open ? .green : .red)

                RectangleMark(
                    x:      .value("Date",  bar.date),
                    yStart: .value("Open",  min(bar.open, bar.close)),
                    yEnd:   .value("Close", max(bar.open, bar.close)),
                    width:  .fixed(bodyWidth)
                )
                .foregroundStyle(bar.close >= bar.open ? .green : .red)
            }
        }
        .chartYScale(domain: yDomain)
        .chartXAxis {
            AxisMarks(values: xAxisValues) { _ in
                AxisGridLine()
                AxisValueLabel(format: .dateTime.month(.abbreviated).day())
            }
        }
        .chartYAxis(.hidden)
    }

    // Sticky right-side Y axis: a tiny chart sharing the same Y domain.
    private var yAxis: some View {
        Chart {
            // Invisible mark to anchor the domain.
            RuleMark(y: .value("anchor", yDomain.lowerBound))
                .foregroundStyle(.clear)
            RuleMark(y: .value("anchor", yDomain.upperBound))
                .foregroundStyle(.clear)
        }
        .chartYScale(domain: yDomain)
        .chartXAxis(.hidden)
        .chartYAxis {
            AxisMarks(position: .trailing) { _ in
                AxisGridLine()
                AxisValueLabel(format: .currency(code: "USD").precision(.fractionLength(0)))
            }
        }
    }

    private var yDomain: ClosedRange<Double> {
        guard let lo = bars.map(\.low).min(),
              let hi = bars.map(\.high).max(),
              lo < hi else {
            return 0...1
        }
        let pad = (hi - lo) * 0.05
        return (lo - pad)...(hi + pad)
    }

    private var xAxisValues: [Date] {
        guard bars.count > 1 else { return [] }
        let step = max(1, bars.count / max(4, Int(CGFloat(bars.count) * slotWidth / 80)))
        return stride(from: 0, to: bars.count, by: step).map { bars[$0].date }
    }
}
