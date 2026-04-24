//
//  BalanceCard.swift
//  PaperTrader
//

import SwiftUI

struct BalanceCard: View {
    @ObservedObject var vm: TraderViewModel
    @State private var showTopUp = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Balance")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(vm.accountBalance, format: .currency(code: "USD"))
                        .font(.title2.bold())
                        .monospacedDigit()
                }
                Spacer()
                Button { showTopUp = true } label: {
                    VStack(spacing: 3) {
                        Image(systemName: "plus.circle").font(.body)
                        Text("Top Up").font(.caption2)
                    }
                    .foregroundStyle(.primary)
                }
            }

            GeometryReader { geo in
                HStack(spacing: 0) {
                    Color.blue.frame(width: geo.size.width * CGFloat(1.0 - vm.position))
                    Color.green
                }
            }
            .frame(height: 4)
            .clipShape(Capsule())

            HStack {
                Circle().fill(.blue).frame(width: 7, height: 7)
                Text("Cash").font(.caption).foregroundStyle(.secondary)
                Text(vm.cashBalance, format: .currency(code: "USD"))
                    .font(.caption.weight(.medium)).monospacedDigit()
                Spacer()
                Circle().fill(.green).frame(width: 7, height: 7)
                Text("Portfolio").font(.caption).foregroundStyle(.secondary)
                Text(vm.portfolioBalance, format: .currency(code: "USD"))
                    .font(.caption.weight(.medium)).monospacedDigit()
            }
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
        .sheet(isPresented: $showTopUp) {
            TopUpSheet(vm: vm, isPresented: $showTopUp)
                .presentationDetents([.height(260)])
                .presentationDragIndicator(.visible)
        }
    }
}

// MARK: - Top Up Sheet

private struct TopUpSheet: View {
    @ObservedObject var vm: TraderViewModel
    @Binding var isPresented: Bool
    @State private var amountText = ""
    @FocusState private var focused: Bool

    private var parsedAmount: Double? {
        Double(amountText.replacingOccurrences(of: ",", with: "."))
            .flatMap { $0 > 0 ? $0 : nil }
    }

    var body: some View {
        VStack(spacing: 20) {
            Text("Top Up Balance")
                .font(.headline)
                .padding(.top, 24)

            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text("$")
                    .font(.system(size: 32, weight: .semibold))
                    .foregroundStyle(.secondary)
                TextField("0", text: $amountText)
                    .font(.system(size: 42, weight: .bold, design: .rounded))
                    .keyboardType(.decimalPad)
                    .multilineTextAlignment(.center)
                    .focused($focused)
                    .monospacedDigit()
            }
            .padding(.horizontal)

            HStack(spacing: 12) {
                Button("Cancel") {
                    isPresented = false
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .background(Color(.systemGray5), in: RoundedRectangle(cornerRadius: 14))
                .foregroundStyle(.primary)

                Button {
                    if let amount = parsedAmount {
                        vm.topUp(amount: amount)
                        isPresented = false
                    }
                } label: {
                    Text("Add \(parsedAmount.map { $0.formatted(.currency(code: "USD")) } ?? "")")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                        .background(parsedAmount != nil ? Color.blue : Color.blue.opacity(0.3),
                                    in: RoundedRectangle(cornerRadius: 14))
                        .foregroundStyle(.white)
                }
                .disabled(parsedAmount == nil)
            }
            .padding(.horizontal)
        }
        .onAppear { focused = true }
    }
}

#Preview {
    BalanceCard(vm: TraderViewModel())
}
