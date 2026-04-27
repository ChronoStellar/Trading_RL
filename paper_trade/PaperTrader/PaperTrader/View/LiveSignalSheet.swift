//
//  LiveSignalSheet.swift
//  PaperTrader
//
//  Modality opened from the Live Signal CTA. Shows rationale for the model's
//  allocation and converts it into a concrete $ amount against the user's budget.
//

import SwiftUI

struct SignalRationale {
    let headline: String
    let bullets: [String]

    /// Derive a short explanation from the price series + the model allocation.
    /// This is best-effort — the model is a black-box; the rationale reflects
    /// *market conditions* the user is about to act in, not exact causal drivers.
    static func build(allocation: Float, bars: [OHLCVBar]) -> SignalRationale {
        let closes = bars.map(\.close)
        var bullets: [String] = []

        if closes.count >= 21 {
            let sma20 = closes.suffix(20).reduce(0, +) / 20
            let pctVsSMA = (closes.last! / sma20 - 1) * 100
            bullets.append(String(format: "Price %.1f%% %@ 20-day MA",
                                  abs(pctVsSMA), pctVsSMA >= 0 ? "above" : "below"))
        }
        if closes.count >= 61 {
            let peak60 = closes.suffix(60).max()!
            let dd = (closes.last! / peak60 - 1) * 100
            bullets.append(String(format: "Drawdown from 60-day high: %.1f%%", dd))
        }
        if closes.count >= 22 {
            let ret20 = (closes.last! / closes[closes.count - 21] - 1) * 100
            bullets.append(String(format: "20-day return: %+.1f%%", ret20))
        }

        let headline: String
        switch allocation {
        case ..<0.3:
            headline = "Defensive — model favors cash."
        case ..<0.6:
            headline = "Mixed — partial exposure."
        default:
            headline = "Constructive — model favors long exposure."
        }
        return SignalRationale(headline: headline, bullets: bullets)
    }
}

struct LiveSignalSheet: View {
    let ticker: String
    let allocation: Float
    let currentPrice: Double
    let rationale: SignalRationale

    @Environment(\.dismiss) private var dismiss
    @State private var budgetText: String = ""

    private var budget: Double { Double(budgetText) ?? 0 }
    private var dollarAlloc: Double { budget * Double(allocation) }
    private var shares: Double {
        guard currentPrice > 0 else { return 0 }
        return dollarAlloc / currentPrice
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {

                    // Allocation readout
                    VStack(alignment: .leading, spacing: 6) {
                        Text("\(ticker) · Today's Signal")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        HStack(alignment: .firstTextBaseline, spacing: 8) {
                            Text("\(Int(allocation * 100))%")
                                .font(.system(size: 56, weight: .black, design: .rounded))
                                .foregroundStyle(allocationColor)
                                .monospacedDigit()
                            Text("of budget")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                    }

                    // Why
                    VStack(alignment: .leading, spacing: 8) {
                        Label("Why this signal", systemImage: "brain.head.profile")
                            .font(.headline)
                        Text(rationale.headline)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        ForEach(rationale.bullets, id: \.self) { b in
                            HStack(alignment: .top, spacing: 8) {
                                Text("•").foregroundStyle(.tertiary)
                                Text(b).font(.footnote).foregroundStyle(.secondary)
                            }
                        }
                        Text("The model outputs a target allocation in [0, 1]. It was trained to match SPY returns while going to cash during drawdowns — not to predict tops.")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                            .padding(.top, 4)
                    }
                    .padding(14)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))

                    // Budget + computed $
                    VStack(alignment: .leading, spacing: 10) {
                        Label("Your budget", systemImage: "dollarsign.circle")
                            .font(.headline)
                        HStack {
                            Text("$")
                                .foregroundStyle(.secondary)
                            TextField("0", text: $budgetText)
                                .keyboardType(.decimalPad)
                                .font(.title3.monospacedDigit())
                        }
                        .padding(12)
                        .background(Color(.systemGray6), in: RoundedRectangle(cornerRadius: 10))

                        Divider()

                        row("Allocate",
                            value: dollarAlloc.formatted(.currency(code: "USD")),
                            tint: allocationColor)
                        row("Keep in cash",
                            value: (budget - dollarAlloc).formatted(.currency(code: "USD")),
                            tint: .primary)
                        if currentPrice > 0 {
                            row("≈ Shares @ \(currentPrice.formatted(.currency(code: "USD")))",
                                value: String(format: "%.4f", shares),
                                tint: .primary)
                        }
                    }
                    .padding(14)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
                }
                .padding()
            }
            .navigationTitle("Signal")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func row(_ label: String, value: String, tint: Color) -> some View {
        HStack {
            Text(label).font(.subheadline).foregroundStyle(.secondary)
            Spacer()
            Text(value).font(.headline.monospacedDigit()).foregroundStyle(tint)
        }
    }

    private var allocationColor: Color {
        switch allocation {
        case ..<0.3: .red
        case ..<0.6: .orange
        default:     .green
        }
    }
}
