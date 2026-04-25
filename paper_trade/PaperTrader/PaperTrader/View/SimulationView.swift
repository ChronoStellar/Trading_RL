//
//  SimulationView.swift
//  PaperTrader
//

import SwiftUI
import Charts

struct SimulationView: View {
    @StateObject private var vm = SimulationViewModel()
    @State private var showRules = false

    private let quickTickers = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]

    var body: some View {
        NavigationStack {
            VStack(spacing: 10) {
                SearchBarRow(text: $vm.searchText, onSubmit: vm.search)
                    .padding(.horizontal)
                    .padding(.top, 8)

                QuickTickerRow(tickers: quickTickers,
                               selected: vm.ticker,
                               onSelect: vm.selectTicker)

                dateRangeCard
                    .padding(.horizontal)

                equityCurveCard
                    .padding(.horizontal)
                    .frame(maxHeight: .infinity)

                Button {
                    vm.simulate()
                } label: {
                    HStack {
                        Image(systemName: vm.isLoading ? "hourglass" : "play.fill")
                        Text(vm.isLoading ? "Simulating…" : "Simulate")
                            .font(.headline)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(.blue, in: RoundedRectangle(cornerRadius: 14))
                    .foregroundStyle(.white)
                }
                .disabled(vm.isLoading)
                .padding(.horizontal)

                HStack {
                    Button { showRules = true } label: {
                        Label("How simulation works", systemImage: "info.circle")
                            .font(.caption)
                    }
                    Spacer()
                }
                .padding(.horizontal)

                if let err = vm.errorMessage {
                    Label(err, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.red)
                        .font(.caption2)
                        .padding(.horizontal)
                }
            }
            .padding(.bottom, 8)
            .navigationTitle("Simulation")
            .navigationBarTitleDisplayMode(.inline)
            .sheet(isPresented: $showRules) {
                RulesSheet().presentationDetents([.medium, .large])
            }
        }
    }

    // MARK: - Date range card

    private var dateRangeCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Range", systemImage: "calendar")
                .font(.subheadline).foregroundStyle(.secondary)
            HStack(spacing: 12) {
                DatePicker("Start", selection: $vm.startDate,
                           in: ...vm.endDate, displayedComponents: .date)
                    .labelsHidden()
                Text("→").foregroundStyle(.tertiary)
                DatePicker("End", selection: $vm.endDate,
                           in: vm.startDate...Date(), displayedComponents: .date)
                    .labelsHidden()
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
    }

    // MARK: - Equity curve (agent vs. buy-and-hold)

    private var equityCurveCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 16) {
                legendDot(.blue,  "Agent")
                legendDot(.gray,  "Buy & Hold")
                Spacer()
                if let r = vm.report, vm.isComplete {
                    Text("\(r.agentReturn >= 0 ? "+" : "")\(String(format: "%.1f%%", r.agentReturn * 100))")
                        .font(.subheadline.bold().monospacedDigit())
                        .foregroundStyle(r.agentReturn >= 0 ? .green : .red)
                }
            }

            if vm.steps.isEmpty {
                ContentUnavailableView(
                    "No simulation yet",
                    systemImage: "chart.line.uptrend.xyaxis",
                    description: Text("Pick a date range and tap Simulate.")
                )
                .frame(maxHeight: .infinity)
            } else {
                let visible = Array(vm.steps.prefix(vm.visibleCount))
                Chart {
                    ForEach(Array(visible.enumerated()), id: \.offset) { _, s in
                        LineMark(x: .value("Date", s.date),
                                 y: .value("Equity", s.equity),
                                 series: .value("Series", "Agent"))
                            .foregroundStyle(.blue)
                        LineMark(x: .value("Date", s.date),
                                 y: .value("Equity", s.benchmark),
                                 series: .value("Series", "Buy & Hold"))
                            .foregroundStyle(.gray.opacity(0.6))
                    }
                }
                .chartYAxis { AxisMarks(position: .trailing) }
                .frame(maxHeight: .infinity)

                if vm.isComplete, let r = vm.report {
                    reportStrip(r)
                }
            }
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
    }

    private func legendDot(_ color: Color, _ label: String) -> some View {
        HStack(spacing: 5) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(label).font(.caption).foregroundStyle(.secondary)
        }
    }

    private func reportStrip(_ r: SimReport) -> some View {
        HStack(spacing: 0) {
            metric("Agent", r.agentReturn.formatted(.percent.precision(.fractionLength(1))),
                   color: r.agentReturn >= 0 ? .green : .red)
            Divider().frame(height: 36)
            metric("B&H", r.benchmarkReturn.formatted(.percent.precision(.fractionLength(1))),
                   color: r.benchmarkReturn >= 0 ? .green : .red)
            Divider().frame(height: 36)
            metric("Sharpe", String(format: "%.2f", r.sharpe),
                   color: r.sharpe >= 1 ? .green : .primary)
            Divider().frame(height: 36)
            metric("Max DD", r.maxDrawdown.formatted(.percent.precision(.fractionLength(1))),
                   color: .red)
        }
        .padding(.top, 6)
    }

    private func metric(_ label: String, _ value: String, color: Color) -> some View {
        VStack(spacing: 2) {
            Text(label).font(.caption2).foregroundStyle(.secondary)
            Text(value).font(.subheadline.bold().monospacedDigit()).foregroundStyle(color)
        }
        .frame(maxWidth: .infinity)
    }
}

#Preview {
    SimulationView()
}
