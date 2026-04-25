//
//  LiveSignalView.swift
//  PaperTrader
//

import SwiftUI

struct LiveSignalView: View {
    @StateObject private var vm = LiveSignalViewModel()
    @State private var showSheet = false

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

                WindowPicker(selected: vm.window, onSelect: vm.selectWindow)
                    .padding(.horizontal)

                if vm.isLoading {
                    ProgressView("Loading \(vm.ticker)…")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if vm.bars.isEmpty {
                    ContentUnavailableView(
                        "No data",
                        systemImage: "chart.xyaxis.line",
                        description: Text("Search a ticker to get started.")
                    )
                    .frame(maxHeight: .infinity)
                } else {
                    StockChartCard(
                        ticker: vm.ticker,
                        bars: vm.displayedBars,
                        currentPrice: vm.currentPrice,
                        lastUpdated: vm.lastUpdated,
                        rangeLabel: vm.window.rawValue
                    )
                    .padding(.horizontal)
                    .frame(maxHeight: .infinity)

                    Button { showSheet = true } label: {
                        HStack {
                            Image(systemName: "sparkles")
                            Text("Today's Signal · \(Int(vm.allocation * 100))%")
                                .font(.headline)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                        .background(allocationColor, in: RoundedRectangle(cornerRadius: 14))
                        .foregroundStyle(.white)
                    }
                    .padding(.horizontal)
                }

                if let err = vm.errorMessage {
                    Label(err, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.red)
                        .font(.caption2)
                        .padding(.horizontal)
                }
            }
            .padding(.bottom, 8)
            .navigationTitle("Live Signal")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button { vm.refresh() } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .disabled(vm.isLoading)
                }
            }
            .sheet(isPresented: $showSheet) {
                LiveSignalSheet(
                    ticker: vm.ticker,
                    allocation: vm.allocation,
                    currentPrice: vm.currentPrice,
                    rationale: SignalRationale.build(allocation: vm.allocation, bars: vm.bars)
                )
                .presentationDetents([.medium, .large])
            }
            .onAppear { vm.refresh() }
        }
    }

    private var allocationColor: Color {
        switch vm.allocation {
        case ..<0.3: .red
        case ..<0.6: .orange
        default:     .green
        }
    }
}

// MARK: - Window picker

private struct WindowPicker: View {
    let selected: DisplayWindow
    let onSelect: (DisplayWindow) -> Void

    var body: some View {
        HStack(spacing: 6) {
            ForEach(DisplayWindow.allCases) { w in
                Button { onSelect(w) } label: {
                    Text(w.rawValue)
                        .font(.caption.weight(.semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 7)
                        .background(
                            selected == w ? Color.blue : Color(.systemGray5),
                            in: RoundedRectangle(cornerRadius: 8)
                        )
                        .foregroundStyle(selected == w ? .white : .primary)
                }
            }
        }
    }
}

#Preview {
    LiveSignalView()
}
