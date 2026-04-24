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
                searchAndCalendarRow
                    .padding(.horizontal)
                    .padding(.top, 8)
                    .padding(.bottom, 10)

                balanceCard
                    .padding(.horizontal)
                    .padding(.bottom, 10)

                quickTickerRow
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
                    stockChartCard
                        .padding(.horizontal)
                        .padding(.bottom, 10)
                        .frame(maxHeight: .infinity)

                    NavigationLink(destination: SignalDetailView(vm: vm)) {
                        aiSignalCard
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
                dateRangeSheet
                    .presentationDetents([.height(200)])
            }
            .onAppear { vm.refresh() }
        }
    }

    // MARK: - Search + Calendar row

    private var searchAndCalendarRow: some View {
        HStack(spacing: 10) {
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                TextField("Search ticker…", text: $vm.searchText)
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

            Button { showCalendar = true } label: {
                Image(systemName: "calendar")
                    .font(.body.weight(.medium))
                    .foregroundStyle(.primary)
                    .frame(width: 42, height: 42)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
            }
        }
    }

    // MARK: - Balance card (cash + portfolio split)

    private var balanceCard: some View {
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
                HStack(spacing: 20) {
                    actionButton("arrow.up.arrow.down", "Pay")
                    actionButton("plus.circle", "Top Up")
                    actionButton("ellipsis.circle", "More")
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
    }

    private func actionButton(_ icon: String, _ label: String) -> some View {
        VStack(spacing: 3) {
            Image(systemName: icon).font(.body)
            Text(label).font(.caption2)
        }
        .foregroundStyle(.primary)
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

    // MARK: - Stock chart card

    private var stockChartCard: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(vm.ticker)
                    .font(.headline.bold())
                Spacer()
                Text(dateRangeLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            CandlestickChartView(bars: vm.bars)

            HStack {
                Text(vm.currentPrice, format: .currency(code: "USD"))
                    .font(.title3.bold())
                    .monospacedDigit()
                if vm.bars.count >= 2 {
                    let chg = (vm.bars.last!.close - vm.bars.first!.close) / vm.bars.first!.close
                    Text(chg, format: .percent.precision(.fractionLength(2)))
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(chg >= 0 ? .green : .red)
                }
                Spacer()
                if let last = vm.lastUpdated {
                    Text(last.formatted(.relative(presentation: .named)))
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
    }

    // MARK: - AI Signal card (tappable → SignalDetailView)

    private var aiSignalCard: some View {
        HStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Image(systemName: "brain.head.profile")
                        .foregroundStyle(.blue)
                    Text("AI Signal")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Text(allocationAction)
                    .font(.title.bold())
                    .foregroundStyle(allocationColor)
                Text(allocationDescription)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 2) {
                Text("\(Int(vm.allocation * 100))%")
                    .font(.system(size: 46, weight: .black, design: .rounded))
                    .foregroundStyle(allocationColor)
                    .monospacedDigit()
                Text("allocate")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Image(systemName: "chevron.right")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.tertiary)
        }
        .padding(16)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    // MARK: - Date range sheet

    private var dateRangeSheet: some View {
        VStack(spacing: 14) {
            Text("Select Range")
                .font(.headline)
                .padding(.top, 20)
            HStack(spacing: 10) {
                ForEach(DateRange.allCases) { range in
                    Button {
                        vm.selectDateRange(range)
                        showCalendar = false
                    } label: {
                        Text(range.rawValue)
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                            .background(
                                vm.dateRange == range ? Color.blue : Color(.systemGray5),
                                in: RoundedRectangle(cornerRadius: 12)
                            )
                            .foregroundStyle(vm.dateRange == range ? .white : .primary)
                    }
                }
            }
            .padding(.horizontal)
            Spacer()
        }
    }

    // MARK: - Helpers

    private var allocationColor: Color {
        switch vm.allocation {
        case ..<0.3: .red
        case ..<0.6: .orange
        default:     .green
        }
    }

    private var allocationAction: String {
        switch vm.allocation {
        case ..<0.3: "SELL"
        case ..<0.6: "HOLD"
        default:     "BUY"
        }
    }

    private var allocationDescription: String {
        switch vm.allocation {
        case ..<0.3: "Stay in cash"
        case ..<0.6: "Partial position"
        default:     "Fully invested"
        }
    }

    private var dateRangeLabel: String {
        guard let first = vm.bars.first, let last = vm.bars.last else {
            return vm.dateRange.rawValue
        }
        let fmt = DateFormatter()
        fmt.dateFormat = "MMM yy"
        return "\(fmt.string(from: first.date)) – \(fmt.string(from: last.date))"
    }
}

#Preview {
    ContentView()
}
