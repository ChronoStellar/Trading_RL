//
//  SearchBarRow.swift
//  PaperTrader
//

import SwiftUI

struct SearchBarRow: View {
    @Binding var text: String
    var onSubmit: () -> Void
    var trailing: AnyView? = nil

    var body: some View {
        HStack(spacing: 10) {
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                TextField("Search ticker…", text: $text)
                    .textInputAutocapitalization(.characters)
                    .autocorrectionDisabled()
                    .submitLabel(.search)
                    .onSubmit(onSubmit)
                if !text.isEmpty {
                    Button { text = "" } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(10)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))

            if let trailing { trailing }
        }
    }
}
