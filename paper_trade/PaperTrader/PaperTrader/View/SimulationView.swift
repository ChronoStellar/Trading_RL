//
//  SimulationView.swift
//  PaperTrader
//

import SwiftUI

struct SimulationView: View {
    @StateObject private var vm = SimulationViewModel()
    @State private var showRules = false
    @State private var pushDetail = false

    private let quickTickers = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]

    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                SearchBarRow(text: $vm.searchText, onSubmit: vm.search)
                    .padding(.horizontal)
                    .padding(.top, 8)

                QuickTickerRow(tickers: quickTickers,
                               selected: vm.ticker,
                               onSelect: vm.selectTicker)

                dateRangeCard
                    .padding(.horizontal)

                Spacer()

                Button {
                    pushDetail = true
                } label: {
                    HStack {
                        Image(systemName: "play.fill")
                        Text("Simulate").font(.headline)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(.blue, in: RoundedRectangle(cornerRadius: 14))
                    .foregroundStyle(.white)
                }
                .padding(.horizontal)

                HStack {
                    Button { showRules = true } label: {
                        Label("How simulation works", systemImage: "info.circle")
                            .font(.caption)
                    }
                    Spacer()
                }
                .padding(.horizontal)
            }
            .padding(.bottom, 8)
            .navigationTitle("Simulation")
            .navigationBarTitleDisplayMode(.inline)
            .navigationDestination(isPresented: $pushDetail) {
                SimulationDetailView(vm: vm)
            }
            .sheet(isPresented: $showRules) {
                RulesSheet().presentationDetents([.medium, .large])
            }
        }
    }

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
}

#Preview {
    SimulationView()
}
