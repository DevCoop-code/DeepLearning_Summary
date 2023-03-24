//
//  ContentView.swift
//  SD2Img2Img
//
//  Created by HanGyo Jeong on 2023/03/23.
//

import SwiftUI

struct ContentView: View {
    @StateObject var imageGenerator = ImageGenerator()
    
    var body: some View {
        VStack {
            ImageToImageView(imageGenerator: imageGenerator)
        }
        .padding()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
