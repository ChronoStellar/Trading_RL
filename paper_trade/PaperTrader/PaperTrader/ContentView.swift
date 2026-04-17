//
//  ContentView.swift
//  PaperTrader
//

import SwiftUI

struct ContentView: View {
    @StateObject private var vm = TraderViewModel()

    var body: some View {
        NavigationStack {
            List {
                // ── Price & signal ────────────────────────────────────────
                Section {
                    HStack {
                        Label("SPY", systemImage: "chart.line.uptrend.xyaxis")
                            .font(.headline)
                        Spacer()
                        Text(vm.currentPrice, format: .currency(code: "USD"))
                            .monospacedDigit()
                    }

                    HStack {
                        Text("Recommended Allocation")
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text(vm.allocation, format: .percent.precision(.fractionLength(1)))
                            .font(.title2.bold())
                            .foregroundStyle(allocationColor)
                    }

                    ProgressView(value: Double(vm.allocation))
                        .tint(allocationColor)
                } header: {
                    Text("Signal")
                }

                // ── Current position ──────────────────────────────────────
                Section {
                    HStack {
                        Text("Position")
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text(vm.position, format: .percent.precision(.fractionLength(1)))
                            .monospacedDigit()
                    }

                    HStack {
                        Text("Unrealised P&L")
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text(vm.unrealisedPnL, format: .percent.precision(.fractionLength(2)))
                            .monospacedDigit()
                            .foregroundStyle(vm.unrealisedPnL >= 0 ? .green : .red)
                    }

                    Button("Apply Allocation") {
                        vm.applyAllocation()
                    }
                    .disabled(vm.isLoading || vm.currentPrice == 0)
                } header: {
                    Text("Portfolio")
                }

                // ── Status ────────────────────────────────────────────────
                if let err = vm.errorMessage {
                    Section {
                        Label(err, systemImage: "exclamationmark.triangle")
                            .foregroundStyle(.red)
                            .font(.footnote)
                    }
                }

                if let updated = vm.lastUpdated {
                    Section {
                        Text("Updated \(updated.formatted(.relative(presentation: .named)))")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("PaperTrader")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        vm.refresh()
                    } label: {
                        if vm.isLoading {
                            ProgressView()
                        } else {
                            Label("Refresh", systemImage: "arrow.clockwise")
                        }
                    }
                    .disabled(vm.isLoading)
                }
            }
            .onAppear { vm.refresh() }
        }
    }

    private var allocationColor: Color {
        switch vm.allocation {
        case ..<0.3:  .red
        case ..<0.6:  .orange
        default:      .green
        }
    }
}

#Preview {
    ContentView()
}
