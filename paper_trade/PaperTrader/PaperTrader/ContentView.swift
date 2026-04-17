//
//  ContentView.swift
//  PaperTrader
//

import SwiftUI

struct ContentView: View {
    @StateObject private var vm = TraderViewModel()

    private let quickTickers = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    searchBar
                    quickTickerRow
                    dateRangePicker

                    if vm.isLoading {
                        ProgressView("Loading \(vm.ticker)…")
                            .frame(height: 260)
                    } else if vm.bars.isEmpty {
                        ContentUnavailableView(
                            "No data",
                            systemImage: "chart.xyaxis.line",
                            description: Text("Search a ticker to get started.")
                        )
                        .frame(height: 260)
                    } else {
                        chartSection
                    }

                    signalSection
                    portfolioSection

                    if let err = vm.errorMessage {
                        Label(err, systemImage: "exclamationmark.triangle")
                            .foregroundStyle(.red)
                            .font(.footnote)
                            .padding(.horizontal)
                    }
                }
                .padding(.vertical, 8)
            }
            .navigationTitle(vm.ticker)
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button { vm.refresh() } label: {
                        Label("Refresh", systemImage: "arrow.clockwise")
                    }
                    .disabled(vm.isLoading)
                }
            }
            .onAppear { vm.refresh() }
        }
    }

    // MARK: - Search bar

    private var searchBar: some View {
        HStack {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
            TextField("Ticker symbol", text: $vm.searchText)
                .textInputAutocapitalization(.characters)
                .autocorrectionDisabled()
                .submitLabel(.search)
                .onSubmit { vm.search() }
            if !vm.searchText.isEmpty {
                Button { vm.searchText = "" } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(10)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }

    // MARK: - Quick ticker chips

    private var quickTickerRow: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(quickTickers, id: \.self) { t in
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

    // MARK: - Date range picker

    private var dateRangePicker: some View {
        Picker("Range", selection: Binding(
            get:  { vm.dateRange },
            set:  { vm.selectDateRange($0) }
        )) {
            ForEach(DateRange.allCases) { range in
                Text(range.rawValue).tag(range)
            }
        }
        .pickerStyle(.segmented)
        .padding(.horizontal)
    }

    // MARK: - Chart

    private var chartSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(vm.currentPrice, format: .currency(code: "USD"))
                    .font(.title2.bold())
                    .monospacedDigit()
                if let last = vm.lastUpdated {
                    Text("· \(last.formatted(.relative(presentation: .named)))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal)

            CandlestickChartView(bars: vm.bars)
                .padding(.horizontal)
        }
    }

    // MARK: - Signal card

    private var signalSection: some View {
        VStack(spacing: 10) {
            HStack {
                Label("AI Signal", systemImage: "brain")
                    .font(.headline)
                Spacer()
                Text(vm.allocation, format: .percent.precision(.fractionLength(1)))
                    .font(.title.bold())
                    .foregroundStyle(allocationColor)
            }
            ProgressView(value: Double(vm.allocation))
                .tint(allocationColor)
            Text(allocationLabel)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
        .padding(.horizontal)
    }

    // MARK: - Portfolio card

    private var portfolioSection: some View {
        VStack(spacing: 10) {
            HStack {
                Label("Portfolio", systemImage: "dollarsign.circle")
                    .font(.headline)
                Spacer()
            }
            Divider()
            HStack {
                Text("Position").foregroundStyle(.secondary)
                Spacer()
                Text(vm.position, format: .percent.precision(.fractionLength(1)))
                    .monospacedDigit()
            }
            HStack {
                Text("Unrealised P&L").foregroundStyle(.secondary)
                Spacer()
                Text(vm.unrealisedPnL, format: .percent.precision(.fractionLength(2)))
                    .monospacedDigit()
                    .foregroundStyle(vm.unrealisedPnL >= 0 ? .green : .red)
            }
            Button(action: { vm.applyAllocation() }) {
                Label("Apply Allocation", systemImage: "checkmark.circle.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(vm.isLoading || vm.currentPrice == 0)
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
        .padding(.horizontal)
    }

    // MARK: - Helpers

    private var allocationColor: Color {
        switch vm.allocation {
        case ..<0.3: .red
        case ..<0.6: .orange
        default:     .green
        }
    }

    private var allocationLabel: String {
        switch vm.allocation {
        case ..<0.3: "Signal: stay in cash"
        case ..<0.6: "Signal: partial position"
        default:     "Signal: fully invested"
        }
    }
}

#Preview {
    ContentView()
}
