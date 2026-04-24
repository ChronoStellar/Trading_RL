//
//  StockChartCard.swift
//  PaperTrader
//

import SwiftUI

struct StockChartCard: View {
    @ObservedObject var vm: TraderViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(vm.ticker)
                    .font(.headline.bold())
                Spacer()
                Text(dateRangeLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            CandlestickChartView(bars: vm.bars)

            HStack {
                Text(vm.currentPrice, format: .currency(code: "USD"))
                    .font(.title3.bold())
                    .monospacedDigit()
                if vm.bars.count >= 2 {
                    let chg = (vm.bars.last!.close - vm.bars.first!.close) / vm.bars.first!.close
                    Text(chg, format: .percent.precision(.fractionLength(2)))
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(chg >= 0 ? .green : .red)
                }
                Spacer()
                if let last = vm.lastUpdated {
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
        guard let first = vm.bars.first, let last = vm.bars.last else {
            return vm.dateRange.rawValue
        }
        let fmt = DateFormatter()
        fmt.dateFormat = "MMM yy"
        return "\(fmt.string(from: first.date)) – \(fmt.string(from: last.date))"
    }
}
