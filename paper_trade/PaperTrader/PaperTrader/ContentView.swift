//
//  ContentView.swift
//  PaperTrader
//

import SwiftUI

struct ContentView: View {
    @StateObject private var vm = TraderViewModel()
    @State private var showCalendar = false

    private let quickTickers = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                SearchBarRow(vm: vm, showCalendar: $showCalendar)
                    .padding(.horizontal)
                    .padding(.top, 8)
                    .padding(.bottom, 10)

                BalanceCard(vm: vm)
                    .padding(.horizontal)
                    .padding(.bottom, 10)

                QuickTickerRow(vm: vm, tickers: quickTickers)
                    .padding(.bottom, 10)

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
                    StockChartCard(vm: vm)
                        .padding(.horizontal)
                        .padding(.bottom, 10)
                        .frame(maxHeight: .infinity)

                    NavigationLink(destination: SignalDetailView(vm: vm)) {
                        AISignalCard(allocation: vm.allocation)
                            .padding(.horizontal)
                    }
                    .buttonStyle(.plain)
                }

                if let err = vm.errorMessage {
                    Label(err, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.red)
                        .font(.caption2)
                        .padding(.horizontal)
                        .padding(.top, 4)
                }
            }
            .padding(.bottom, 8)
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button { vm.refresh() } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .disabled(vm.isLoading)
                }
            }
            .sheet(isPresented: $showCalendar) {
                DateRangeSheet(vm: vm, isPresented: $showCalendar)
                    .presentationDetents([.height(200)])
            }
            .onAppear { vm.refresh() }
        }
    }
}

#Preview {
    ContentView()
}
