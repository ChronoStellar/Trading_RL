//
//  SearchBarRow.swift
//  PaperTrader
//

import SwiftUI

struct SearchBarRow: View {
    @ObservedObject var vm: TraderViewModel
    @Binding var showCalendar: Bool

    var body: some View {
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
}
