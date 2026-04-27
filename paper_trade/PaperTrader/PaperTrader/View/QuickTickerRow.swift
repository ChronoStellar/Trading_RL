//
//  QuickTickerRow.swift
//  PaperTrader
//

import SwiftUI

struct QuickTickerRow: View {
    let tickers: [String]
    let selected: String
    var onSelect: (String) -> Void

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(tickers, id: \.self) { t in
                    Button(t) { onSelect(t) }
                        .buttonStyle(.bordered)
                        .tint(selected == t ? .blue : .secondary)
                        .controlSize(.small)
                }
            }
            .padding(.horizontal)
        }
    }
}
