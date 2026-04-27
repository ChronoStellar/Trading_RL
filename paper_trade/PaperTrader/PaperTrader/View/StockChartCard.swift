//
//  StockChartCard.swift
//  PaperTrader
//

import SwiftUI

struct StockChartCard: View {
    let ticker: String
    let bars: [OHLCVBar]
    let currentPrice: Double
    let lastUpdated: Date?
    let rangeLabel: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(ticker).font(.headline.bold())
                Spacer()
                Text(dateRangeLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            CandlestickChartView(bars: bars)

            HStack {
                Text(currentPrice, format: .currency(code: "USD"))
                    .font(.title3.bold())
                    .monospacedDigit()
                if bars.count >= 2 {
                    let chg = (bars.last!.close - bars.first!.close) / bars.first!.close
                    Text(chg, format: .percent.precision(.fractionLength(2)))
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(chg >= 0 ? .green : .red)
                }
                Spacer()
                if let last = lastUpdated {
                    Text(last.formatted(.relative(presentation: .named)))
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
    }

    private var dateRangeLabel: String {
        guard let first = bars.first, let last = bars.last else { return rangeLabel }
        let fmt = DateFormatter()
        fmt.dateFormat = "MMM yy"
        return "\(fmt.string(from: first.date)) – \(fmt.string(from: last.date))"
    }
}
