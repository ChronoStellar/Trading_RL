//
//  DateRangeSheet.swift
//  PaperTrader
//

import SwiftUI

struct DateRangeSheet: View {
    @ObservedObject var vm: TraderViewModel
    @Binding var isPresented: Bool

    var body: some View {
        VStack(spacing: 14) {
            Text("Select Range")
                .font(.headline)
                .padding(.top, 20)
            HStack(spacing: 10) {
                ForEach(DateRange.allCases) { range in
                    Button {
                        vm.selectDateRange(range)
                        isPresented = false
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
}
