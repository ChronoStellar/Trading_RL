//
//  ContentView.swift
//  PaperTrader
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            LiveSignalView()
                .tabItem { Label("Live Signal", systemImage: "sparkles") }

            SimulationView()
                .tabItem { Label("Simulation", systemImage: "play.circle") }
        }
    }
}

#Preview {
    ContentView()
}
