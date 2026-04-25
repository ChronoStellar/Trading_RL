//
//  SimulationDetailView.swift
//  PaperTrader
//
//  Pushed from SimulationView. Plays the simulation bar-by-bar; pauses at every
//  critical decision point and asks the user to Buy / Sell / Hold. The user's
//  choices drive the equity curve.
//

import SwiftUI

struct SimulationDetailView: View {
    @ObservedObject var vm: SimulationViewModel
    @State private var didStart = false
    @State private var inspectAction: UserAction?

    var body: some View {
        VStack(spacing: 0) {
            chartSection
                .padding(.horizontal)
                .padding(.top, 8)
                .frame(maxHeight: .infinity)

            statusBar
                .padding(.horizontal)
                .padding(.top, 8)

            if vm.isComplete, let report = vm.report {
                reportCard(report)
                    .padding(.horizontal)
                    .padding(.top, 10)
                    .padding(.bottom, 12)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .navigationTitle("\(vm.ticker) · Simulation")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            guard !didStart else { return }
            didStart = true
            await vm.startSimulation()
        }
        .onDisappear { vm.cancelPlayback() }
        .sheet(item: $vm.pendingDecision) { decision in
            DecisionSheet(
                decision: decision,
                userPosition: vm.userPosition,
                onChoice: { vm.decide($0) }
            )
            .presentationDetents([.height(340)])
            .interactiveDismissDisabled()
        }
        .sheet(item: $inspectAction) { action in
            UserActionSheet(action: action, ticker: vm.ticker)
                .presentationDetents([.height(280)])
        }
    }

    // MARK: - Chart

    private var chartSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                legendDot(.green, "Price")
                legendDot(.blue,  "You")
                legendDot(.gray,  "B&H")
                Spacer()
                if !vm.steps.isEmpty {
                    Text("Bar \(vm.revealedIndex + 1) / \(vm.steps.count)")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }

            ZStack {
                if vm.isLoading {
                    ProgressView("Preparing simulation…")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if let err = vm.errorMessage {
                    Label(err, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.red)
                        .font(.callout)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if !vm.steps.isEmpty {
                    GeometryReader { geo in
                        ZStack {
                            gridPath(in: geo.size)
                                .stroke(Color.primary.opacity(0.07), lineWidth: 1)

                            // Price line (revealed portion only)
                            pricePath(in: geo.size)
                                .stroke(.green,
                                        style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))

                            // User actions placed at execution bars
                            ForEach(vm.userActions) { action in
                                userMarker(action, in: geo.size)
                            }

                            // Pulse the bar at a pending decision
                            if let pending = vm.pendingDecision {
                                pulseMarker(at: pending.index, in: geo.size)
                            }
                        }
                        .animation(.linear(duration: 0.05), value: vm.revealedIndex)
                    }
                    .frame(maxHeight: .infinity)
                }
            }
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private var statusBar: some View {
        HStack(spacing: 14) {
            statBlock("Position", "\(Int(vm.userPosition * 100))%",
                      color: vm.userPosition > 0 ? .blue : .secondary)
            Divider().frame(height: 28)
            if let last = vm.userEquity.last, let first = vm.userEquity.first {
                let r = last / first - 1
                statBlock("P&L",
                          (r).formatted(.percent.precision(.fractionLength(2))),
                          color: r >= 0 ? .green : .red)
            }
            Divider().frame(height: 28)
            statBlock("Actions", "\(vm.userActions.count) / \(vm.actions.count)",
                      color: .primary)
        }
        .padding(10)
        .frame(maxWidth: .infinity)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private func statBlock(_ label: String, _ value: String, color: Color) -> some View {
        VStack(spacing: 2) {
            Text(label).font(.caption2).foregroundStyle(.secondary)
            Text(value).font(.subheadline.bold().monospacedDigit()).foregroundStyle(color)
        }
        .frame(maxWidth: .infinity)
    }

    private func legendDot(_ color: Color, _ label: String) -> some View {
        HStack(spacing: 5) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(label).font(.caption).foregroundStyle(.secondary)
        }
    }

    // MARK: - Path / markers

    private func gridPath(in size: CGSize) -> Path {
        Path { path in
            for frac in [0.25, 0.5, 0.75] as [CGFloat] {
                let y = size.height * (1 - frac)
                path.move(to: CGPoint(x: 0, y: y))
                path.addLine(to: CGPoint(x: size.width, y: y))
            }
        }
    }

    /// Bounds use the *full* price range so the y-axis doesn't jump as bars reveal.
    private func priceBounds() -> (min: Double, max: Double)? {
        let prices = vm.steps.map(\.price)
        guard let lo = prices.min(), let hi = prices.max(), hi > lo else { return nil }
        return (lo, hi)
    }

    private func xPosition(forIndex i: Int, in size: CGSize) -> CGFloat {
        guard vm.steps.count > 1 else { return 0 }
        return CGFloat(i) / CGFloat(vm.steps.count - 1) * size.width
    }

    private func yPosition(forPrice p: Double, in size: CGSize) -> CGFloat {
        guard let b = priceBounds() else { return size.height / 2 }
        let range = b.max - b.min
        return CGFloat((b.max - p) / range) * size.height
    }

    private func pricePath(in size: CGSize) -> Path {
        guard priceBounds() != nil else { return Path() }
        var path = Path()
        for i in 0...vm.revealedIndex {
            let x = xPosition(forIndex: i, in: size)
            let y = yPosition(forPrice: vm.steps[i].price, in: size)
            if i == 0 { path.move(to: CGPoint(x: x, y: y)) }
            else      { path.addLine(to: CGPoint(x: x, y: y)) }
        }
        return path
    }

    @ViewBuilder
    private func userMarker(_ action: UserAction, in size: CGSize) -> some View {
        if action.stepIndex < vm.steps.count {
            let x = xPosition(forIndex: action.stepIndex, in: size)
            let y = yPosition(forPrice: action.price, in: size)
            let (icon, color): (String, Color) = {
                switch action.choice {
                case .buy:  return ("arrow.up.circle.fill", .blue)
                case .sell: return ("arrow.down.circle.fill", .red)
                case .hold: return ("equal.circle.fill", .gray)
                }
            }()
            Button { inspectAction = action } label: {
                Image(systemName: icon)
                    .font(.title3)
                    .foregroundStyle(color)
                    .background(Circle().fill(.background).padding(2))
            }
            .position(x: x, y: y)
        }
    }

    @ViewBuilder
    private func pulseMarker(at index: Int, in size: CGSize) -> some View {
        if index < vm.steps.count {
            let x = xPosition(forIndex: index, in: size)
            let y = yPosition(forPrice: vm.steps[index].price, in: size)
            PulseDot(color: .orange).position(x: x, y: y)
        }
    }

    // MARK: - Report

    private func reportCard(_ r: SimReport) -> some View {
        VStack(spacing: 12) {
            Text("Report")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            Divider()
            HStack(spacing: 0) {
                metric("You",
                       r.userReturn.formatted(.percent.precision(.fractionLength(1))),
                       color: r.userReturn >= 0 ? .green : .red)
                Divider().frame(height: 48)
                metric("Model",
                       r.modelReturn.formatted(.percent.precision(.fractionLength(1))),
                       color: r.modelReturn >= 0 ? .green : .red)
                Divider().frame(height: 48)
                metric("B&H",
                       r.benchmarkReturn.formatted(.percent.precision(.fractionLength(1))),
                       color: r.benchmarkReturn >= 0 ? .green : .red)
                Divider().frame(height: 48)
                metric("Decisions",
                       "\(r.userActions) / \(r.modelActions)",
                       color: .primary)
            }
        }
        .padding(16)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private func metric(_ label: String, _ value: String, color: Color) -> some View {
        VStack(spacing: 5) {
            Text(label).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.title3.bold().monospacedDigit()).foregroundStyle(color)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Pulse animation

private struct PulseDot: View {
    let color: Color
    @State private var scale: CGFloat = 1
    var body: some View {
        ZStack {
            Circle().fill(color.opacity(0.25))
                .frame(width: 28, height: 28)
                .scaleEffect(scale)
            Circle().fill(color).frame(width: 10, height: 10)
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true)) {
                scale = 1.8
            }
        }
    }
}

// MARK: - Decision sheet (modal during pause)

private struct DecisionSheet: View {
    let decision: ActionPoint
    let userPosition: Double
    let onChoice: (UserChoice) -> Void
    @Environment(\.dismiss) private var dismiss

    private var target:    Double { Double(decision.modelAlloc) }
    private var afterBuy:  Double { max(userPosition, target) }
    private var afterSell: Double { min(userPosition, target) }
    /// Buy is only meaningful when the model wants more exposure than the user holds.
    private var canBuy:  Bool { decision.kind == .buy  && target > userPosition }
    private var canSell: Bool { decision.kind == .sell && target < userPosition }

    var body: some View {
        VStack(spacing: 16) {
            HStack {
                Image(systemName: decision.kind == .buy ? "arrow.up.circle.fill" : "arrow.down.circle.fill")
                    .font(.system(size: 28))
                    .foregroundStyle(decision.kind == .buy ? .blue : .red)
                VStack(alignment: .leading) {
                    Text("Critical decision")
                        .font(.caption).foregroundStyle(.secondary)
                    Text(decision.date.formatted(date: .abbreviated, time: .omitted))
                        .font(.headline)
                }
                Spacer()
                Text(decision.price.formatted(.currency(code: "USD")))
                    .font(.title3.bold().monospacedDigit())
            }
            .padding(.horizontal)
            .padding(.top, 18)

            VStack(alignment: .leading, spacing: 6) {
                Text("Model suggests \(decision.kind == .buy ? "BUY" : "SELL") · target \(Int(decision.modelAlloc * 100))%")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                HStack(spacing: 6) {
                    Text("Your position:").font(.caption).foregroundStyle(.tertiary)
                    Text("\(Int(userPosition * 100))%")
                        .font(.caption.bold().monospacedDigit())
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal)

            HStack(spacing: 10) {
                choiceButton("Sell",
                             sub: "→ \(Int(afterSell * 100))%",
                             color: .red, systemImage: "arrow.down",
                             enabled: canSell) { onChoice(.sell) }
                choiceButton("Hold",
                             sub: "→ \(Int(userPosition * 100))%",
                             color: .gray, systemImage: "equal",
                             enabled: true) { onChoice(.hold) }
                choiceButton("Buy",
                             sub: "→ \(Int(afterBuy * 100))%",
                             color: .blue, systemImage: "arrow.up",
                             enabled: canBuy) { onChoice(.buy) }
            }
            .padding(.horizontal)

            Text("Buy / Sell aligns your position with the model's target (\(Int(target * 100))%).")
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            Spacer()
        }
    }

    private func choiceButton(_ label: String, sub: String, color: Color,
                              systemImage: String, enabled: Bool,
                              action: @escaping () -> Void) -> some View {
        Button {
            action()
            dismiss()
        } label: {
            VStack(spacing: 6) {
                Image(systemName: systemImage).font(.title2)
                Text(label).font(.subheadline.bold())
                Text(sub).font(.caption2.monospacedDigit()).opacity(0.85)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
            .background(color.opacity(enabled ? 1.0 : 0.35),
                        in: RoundedRectangle(cornerRadius: 12))
            .foregroundStyle(.white)
        }
        .disabled(!enabled)
    }
}

// MARK: - User action inspector

private struct UserActionSheet: View {
    let action: UserAction
    let ticker: String
    @Environment(\.dismiss) private var dismiss

    private var label: String {
        switch action.choice { case .buy: "BUY"; case .sell: "SELL"; case .hold: "HOLD" }
    }
    private var color: Color {
        switch action.choice { case .buy: .blue; case .sell: .red; case .hold: .gray }
    }

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Text(label).font(.title2.bold()).foregroundStyle(color)
                    Spacer()
                    Text("\(ticker) · \(action.date.formatted(date: .abbreviated, time: .omitted))")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Divider()
                row("Price", action.price.formatted(.currency(code: "USD")))
                row("Position before", "\(Int(action.positionBefore * 100))%")
                row("Position after",  "\(Int(action.positionAfter  * 100))%")
                Spacer()
            }
            .padding()
            .navigationTitle("Your action")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).font(.subheadline).foregroundStyle(.secondary)
            Spacer()
            Text(value).font(.subheadline.bold().monospacedDigit())
        }
    }
}
