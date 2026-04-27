//
//  RulesSheet.swift
//  PaperTrader
//

import SwiftUI

struct RulesSheet: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    section(
                        "How the simulation works",
                        "Each trading day, the model reads the last ~60 days of price and technical features, outputs a target allocation in [0, 1], and the portfolio rebalances to that allocation. We step through the selected date range one bar at a time."
                    )
                    section(
                        "Starting capital",
                        "Both the agent and the Buy & Hold benchmark start at 1.0. All curves are normalized — the shape is what matters, not the dollar amount."
                    )
                    section(
                        "Action space",
                        "Long-only. 0 = all cash, 1 = fully long. No shorting, no leverage. This is a deliberate design choice — shorting is not safely modelable at retail scale."
                    )
                    section(
                        "Costs",
                        "Each rebalance incurs a transaction cost (5 bps) plus slippage (5 bps), applied to the change in position size (|Δposition|). Staying put is free."
                    )
                    section(
                        "Benchmark",
                        "Buy & Hold on the same ticker over the same window. The model's goal is to match benchmark return with a better Sharpe — mainly by going to cash during drawdowns."
                    )
                    section(
                        "What this is not",
                        "A trading recommendation. Past simulated performance does not predict future returns. The agent has never placed a real order."
                    )
                }
                .padding()
            }
            .navigationTitle("Rules")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func section(_ title: String, _ body: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.headline)
            Text(body).font(.subheadline).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
