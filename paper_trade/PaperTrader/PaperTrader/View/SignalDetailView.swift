//
//  SignalDetailView.swift
//  PaperTrader
//
//  Tapped from AI Signal card. Plays an animated price-return chart
//  then slides up a report card (profit, Sharpe, drawdown).
//

import SwiftUI

struct SignalDetailView: View {
    @ObservedObject var vm: TraderViewModel

    @State private var drawProgress: CGFloat = 0
    @State private var showReport = false

    var body: some View {
        VStack(spacing: 0) {
            performanceSection
                .padding(.horizontal)
                .padding(.top, 8)
                .frame(maxHeight: .infinity)

            if showReport, let report = vm.portfolioReport {
                reportCard(report)
                    .padding(.horizontal)
                    .padding(.bottom, 16)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .navigationTitle("\(vm.ticker) · Performance")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            drawProgress = 0
            showReport = false
            withAnimation(.easeInOut(duration: 2.4)) {
                drawProgress = 1.0
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                withAnimation(.spring(response: 0.45)) {
                    showReport = true
                }
            }
        }
    }

    // MARK: - Animated performance chart

    private var performanceSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                legendDot(.green, "Buy & Hold")
                Spacer()
                Text(vm.dateRange.rawValue)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            // Animated line chart drawn via Path.trim
            ZStack(alignment: .bottom) {
                GeometryReader { geo in
                    ZStack {
                        gridPath(in: geo.size)
                            .stroke(Color.primary.opacity(0.07), lineWidth: 1)

                        pricePath(in: geo.size)
                            .trim(from: 0, to: drawProgress)
                            .stroke(
                                .green,
                                style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round)
                            )
                    }
                }
                .frame(maxHeight: .infinity)

                // Allocation bar at chart bottom
                allocationBar
                    .frame(height: 22)
                    .opacity(Double(drawProgress))
            }

            // X-axis date labels
            if vm.bars.count >= 2 {
                HStack {
                    Text(vm.bars.first!.date, style: .date)
                    Spacer()
                    Text(vm.bars.last!.date, style: .date)
                }
                .font(.caption2)
                .foregroundStyle(.tertiary)
            }
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private func legendDot(_ color: Color, _ label: String) -> some View {
        HStack(spacing: 5) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(label).font(.caption).foregroundStyle(.secondary)
        }
    }

    private func gridPath(in size: CGSize) -> Path {
        Path { path in
            for frac in [0.25, 0.5, 0.75] as [CGFloat] {
                let y = size.height * (1 - frac)
                path.move(to: CGPoint(x: 0, y: y))
                path.addLine(to: CGPoint(x: size.width, y: y))
            }
        }
    }

    private func pricePath(in size: CGSize) -> Path {
        let closes = vm.bars.map { $0.close }
        guard closes.count > 1,
              let minC = closes.min(), let maxC = closes.max(),
              minC < maxC else { return Path() }
        let range = maxC - minC
        var path = Path()
        for (i, c) in closes.enumerated() {
            let x = CGFloat(i) / CGFloat(closes.count - 1) * size.width
            let y = CGFloat((maxC - c) / range) * size.height
            if i == 0 { path.move(to: CGPoint(x: x, y: y)) }
            else { path.addLine(to: CGPoint(x: x, y: y)) }
        }
        return path
    }

    private var allocationBar: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.primary.opacity(0.06))
                RoundedRectangle(cornerRadius: 4)
                    .fill(allocationColor.opacity(0.55))
                    .frame(width: geo.size.width * CGFloat(vm.allocation))
                Text("Allocation · \(Int(vm.allocation * 100))%")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .padding(.leading, 8)
            }
        }
    }

    // MARK: - Report card

    private func reportCard(_ report: PortfolioReport) -> some View {
        VStack(spacing: 12) {
            Text("Report · \(vm.dateRange.rawValue)")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)

            Divider()

            HStack(spacing: 0) {
                metric(
                    "Profit",
                    report.periodReturn.formatted(.percent.precision(.fractionLength(1))),
                    color: report.periodReturn >= 0 ? .green : .red
                )
                Divider().frame(height: 48)
                metric(
                    "Sharpe",
                    String(format: "%.2f", report.sharpeRatio),
                    color: report.sharpeRatio >= 1 ? .green : report.sharpeRatio >= 0 ? .primary : .red
                )
                Divider().frame(height: 48)
                metric(
                    "Max DD",
                    report.maxDrawdown.formatted(.percent.precision(.fractionLength(1))),
                    color: .red
                )
            }

            Button(action: { vm.applyAllocation() }) {
                Label("Apply \(Int(vm.allocation * 100))% Allocation", systemImage: "checkmark.circle.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(allocationColor)
            .disabled(vm.isLoading || vm.currentPrice == 0)
        }
        .padding(16)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private func metric(_ label: String, _ value: String, color: Color) -> some View {
        VStack(spacing: 5) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.title3.bold())
                .foregroundStyle(color)
                .monospacedDigit()
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Helpers

    private var allocationColor: Color {
        switch vm.allocation {
        case ..<0.3: .red
        case ..<0.6: .orange
        default:     .green
        }
    }
}

#Preview {
    NavigationStack {
        SignalDetailView(vm: TraderViewModel())
    }
}
