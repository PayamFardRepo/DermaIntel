import React, {useState} from 'react';
import { View, Text, StyleSheet, TextInput, Button, Pressable, Image } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { useRouter } from 'expo-router';
import * as ImagePicker from 'expo-image-picker';
import { Alert } from "react-native";
import { Linking } from 'react-native';

export default function PhotoScreen() {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [modelPreds, setModelPreds] = useState<string>("Waiting...");

  //Choose method of uploading picture
  const uploadPhotoPress = () => {
    Alert.alert(
      "Upload Photo",
      "Choose a source",
      [
        {text: "Camera", onPress: cameraPhoto},
        {text: "Library", onPress: libraryPhoto},
        {text: "Cancel", style: "cancel"},
      ]
    );
  };

  //Passes data to backend and receives classification
  const classifyPhoto = async () => {
    if (!imageUri) return alert("Please upload a photo first.");

    // Create form data
    const formData = new FormData();
    formData.append("file", {
      uri: imageUri,
      name: "photo.jpg",
      type: "image/jpeg",
    } as any);

    try {
      const response = await fetch("http://192.168.68.61:8000/upload/", {
        method: "POST",
        headers: {
          "Content-Type": "multipart/form-data",
        },
        body: formData,
      });

      const binary_data = await response.json();
      console.log("Binary response:", binary_data);

      if (binary_data.confidence_boolean === true) {
        await runFullClassify(formData);
      } else {
        Alert.alert(
          "Low Model Confidence",
          "This looks like a non-lesion image. Do you want to continue?",
          [
            { text: "Cancel", style: "cancel" },
            { text: "Continue", onPress: () => runFullClassify(formData) },
          ]
        );
      }
    } catch (error) {
      console.error("Upload failed:", error);
      alert("Upload failed!");
    }
  };

  //Gets camera permission and updates imageUri
  const cameraPhoto = async () => {
    const { granted } = await ImagePicker.requestCameraPermissionsAsync();

    if (!granted) {
      alert("Camera permission denied.");
      return;
    }

    const result = await ImagePicker.launchCameraAsync();
    if (!result.canceled) {
      const photo = result.assets[0];
      setModelPreds("Waiting...");
      setImageUri(photo.uri)
      console.log("Camera photo:", photo);
    }
  };

  //Gets library permission and updates imageUri
  const libraryPhoto = async () => {
    const { granted } = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!granted) {
      alert("Library permission denied.");
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync();
    if (!result.canceled) {
      const photo = result.assets[0];
      setModelPreds("Waiting...");
      setImageUri(photo.uri);
      console.log("Library photo:", photo);
    }
  };

  //Classifies under 7 lesion types
  const runFullClassify = async (formData: FormData) => {
    try {
      const response = await fetch("http://192.168.68.61:8000/full_classify/", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      console.log("Full classifier response: ", result);
      
      const probsText = Object.entries(result.probabilities)
        .map(
          ([key, value]: [string, number]) =>
            `  ${result.key_map?.[key] ?? key}: ${(value * 100).toFixed(1)}%`
        )
        .join("\n");

      const displayText = `Predicted: ${result.predicted_class} (${(result.lesion_confidence * 100).toFixed(1)}%)\n\nProbabilities:\n${probsText}`;

      setModelPreds(displayText);
      
    } catch (error) {
      console.error("Full classify failed:", error);
      alert("Full classify failed!");
    }
  }
   
  return (
    <View style={styles.container}>
      {imageUri && (
        <Image
          source={{uri: imageUri}}
          style = {styles.image}
          resizeMode = "contain"
        />
      )}

      <Pressable style = {styles.button} onPress={uploadPhotoPress}>
        <Text style = {styles.buttonText}>Upload Photo</Text>
      </Pressable>

      <Pressable style = {styles.button} onPress={classifyPhoto}>
        <Text style = {styles.buttonText}>Run</Text>
      </Pressable>

      <Text style = {styles.preds}>{modelPreds}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#fff",
  },
  button: {
    backgroundColor: "#8d8f92ff",
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 8,
    marginBottom: 10,
  },
  buttonText: {
    color: "white",
    fontSize: 18,
  },
  image: {
    width: 350,
    height: 350,
    marginTop: 70,
    marginBottom: 15,
  },
  preds: {
    paddingVertical: 25,
    fontSize: 16
  },
});