//
//  AISignalCard.swift
//  PaperTrader
//

import SwiftUI

struct AISignalCard: View {
    let allocation: Float

    var body: some View {
        HStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Image(systemName: "brain.head.profile")
                        .foregroundStyle(.blue)
                    Text("AI Signal")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Text(action)
                    .font(.title.bold())
                    .foregroundStyle(color)
                Text(description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 2) {
                Text("\(Int(allocation * 100))%")
                    .font(.system(size: 46, weight: .black, design: .rounded))
                    .foregroundStyle(color)
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

    private var color: Color {
        switch allocation {
        case ..<0.3: .red
        case ..<0.6: .orange
        default:     .green
        }
    }

    private var action: String {
        switch allocation {
        case ..<0.3: "SELL"
        case ..<0.6: "HOLD"
        default:     "BUY"
        }
    }

    private var description: String {
        switch allocation {
        case ..<0.3: "Stay in cash"
        case ..<0.6: "Partial position"
        default:     "Fully invested"
        }
    }
}

#Preview {
    AISignalCard(allocation: 0.72)
        .padding()
}
