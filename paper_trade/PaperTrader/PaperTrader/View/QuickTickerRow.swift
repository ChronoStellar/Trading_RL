//
//  QuickTickerRow.swift
//  PaperTrader
//

import SwiftUI

struct QuickTickerRow: View {
    @ObservedObject var vm: TraderViewModel
    let tickers: [String]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(tickers, id: \.self) { t in
                    Button(t) {
                        vm.searchText = t
                        vm.search()
                    }
                    .buttonStyle(.bordered)
                    .tint(vm.ticker == t ? .blue : .secondary)
                    .controlSize(.small)
                }
            }
            .padding(.horizontal)
        }
    }
}
