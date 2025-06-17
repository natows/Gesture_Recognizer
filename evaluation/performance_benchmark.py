# evaluation/performance_benchmark.py
import time
import psutil
import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class ModelBenchmark:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def get_model_memory_usage(self, model_path, model_type='pickle'):
        """Zmierz rzeczywiste zużycie pamięci modelu"""
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  
        
        if model_type == 'pickle':
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        elif model_type == 'keras':
            model = load_model(model_path)
        
        memory_after = process.memory_info().rss / 1024 / 1024  
        memory_usage = memory_after - memory_before
        
        return model, memory_usage
    
    def measure_inference_time(self, model, input_data, n_runs=100, model_type='sklearn'):
        """Zmierz rzeczywisty czas inferencji"""
        
        for _ in range(10):
            try:
                if model_type == 'sklearn':
                    _ = model.predict(input_data)
                elif model_type == 'keras':
                    _ = model.predict(input_data, verbose=0)
            except:
                pass
        
        times = []
        for _ in range(n_runs):
            start_time = time.perf_counter()
            
            try:
                if model_type == 'sklearn':
                    prediction = model.predict(input_data)
                elif model_type == 'keras':
                    prediction = model.predict(input_data, verbose=0)
            except Exception as e:
                return {'error': str(e)}
                
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times)
        }
    
    def prepare_test_data(self):
        """Przygotuj dane testowe dla wszystkich modeli"""
        
        landmarks_array = np.random.random(42)
        
        test_data = {
            'landmarks': landmarks_array,  
            'image_224': np.random.random((1, 224, 224, 3)),  
            'image_64': np.random.random((1, 64, 64, 3)), 
            'sequence': np.random.random((1, 10, 42))  
        }
        
        return test_data
    
    def benchmark_all_models(self):
        """Zmierz wszystkie modele"""
        
        print(" Rozpoczynam benchmark wszystkich modeli...")
        test_data = self.prepare_test_data()
        results = []
        
        models_config = [
            {
                'name': 'kNN',
                'path': '../models/kNN/kNN.pkl',
                'type': 'pickle',
                'sklearn_type': True,
                'input_data': test_data['landmarks'].reshape(1, -1)
            },
            {
                'name': 'DTC',
                'path': '../models/DTC/DTC_extended.pkl',
                'type': 'pickle', 
                'sklearn_type': True,
                'input_data': test_data['landmarks'].reshape(1, -1)
            },
            {
                'name': 'Naive Bayes',
                'path': '../models/NB/NaiveBayes_model.pkl',
                'type': 'pickle',
                'sklearn_type': True, 
                'input_data': test_data['landmarks'].reshape(1, -1)
            },
            {
                'name': 'MLP',
                'path': '../models/MLP/MLP.pkl',
                'type': 'pickle',
                'sklearn_type': True,
                'input_data': test_data['landmarks'].reshape(1, -1)
            },
            {
                'name': 'CNN', 
                'path': '../models/CNN/cnn_custom2_only_model.keras',
                'type': 'keras',
                'sklearn_type': False,
                'input_data': test_data['image_64']  
            },
            {
                'name': 'MobileNetV2',
                'path': '../models/MobileNetV2/mobilenetv2_extended_model.keras',
                'type': 'keras',
                'sklearn_type': False,
                'input_data': test_data['image_224']
            },
            {
                'name': 'LSTM',
                'path': '../models/LSTM/lstm_gesture_model.keras',
                'type': 'keras',
                'sklearn_type': False,
                'input_data': test_data['sequence']
            }

        ]
        
        for config in models_config:
            print(f"\n Testowanie modelu: {config['name']}")
            
            try:
                model, memory_usage = self.get_model_memory_usage(
                    config['path'], config['type']
                )
                
                model_type = 'sklearn' if config['sklearn_type'] else 'keras'
                timing_stats = self.measure_inference_time(
                    model, config['input_data'], n_runs=100, model_type=model_type
                )
                
                if 'error' in timing_stats:
                    raise Exception(timing_stats['error'])
                
                results.append({
                    'Model': config['name'],
                    'Memory (MB)': round(memory_usage, 2),
                    'Mean Time (ms)': round(timing_stats['mean_time'], 3),
                    'Std Time (ms)': round(timing_stats['std_time'], 3),
                    'Min Time (ms)': round(timing_stats['min_time'], 3),
                    'Max Time (ms)': round(timing_stats['max_time'], 3)
                })
                
                print(f"  Pamięć: {memory_usage:.2f} MB")
                print(f"   Czas: {timing_stats['mean_time']:.3f} ± {timing_stats['std_time']:.3f} ms")
                
            except Exception as e:
                print(f"  Błąd: {e}")
                results.append({
                    'Model': config['name'],
                    'Memory (MB)': 'Error',
                    'Mean Time (ms)': 'Error',
                    'Std Time (ms)': 'Error',
                    'Min Time (ms)': 'Error',
                    'Max Time (ms)': 'Error'
                })
        
        return results
    
    def create_comparison_table(self, benchmark_results):
        """Stwórz comprehensive comparison table"""
        
        accuracy_data = {
            'kNN': 99.0,
            'DTC': 99.2, 
            'Naive Bayes': 97.8,
            'MLP': 99.8,
            'CNN': 99.6,  
            'MobileNetV2': 99.9,
            'LSTM': 98.4
        }
        
        practical_results = {
            'kNN': '18/20 ✓',
            'DTC': '20/20 ✓✓✓',
            'Naive Bayes': '13/20',
            'MLP': '17/20 ✓', 
            'CNN': '10/20 ✓✓',
            'MobileNetV2': '15/20 ✓',
            'LSTM': 'Overfitted'
        }
        

        final_results = []
    
        
        for result in benchmark_results:
            model_name = result['Model']
            if result['Mean Time (ms)'] != 'Error':
                time_str = f"{result['Mean Time (ms)']} ± {result['Std Time (ms)']}"
                memory_str = str(result['Memory (MB)'])
            else:
                time_str = 'Error'
                memory_str = 'Error'
                
            final_results.append({
                'Model': model_name,
                'Test Accuracy (%)': accuracy_data.get(model_name, 'N/A'),
                'Inference Time (ms)': time_str,
                'Memory Usage (MB)': memory_str,
                'Practical Test': practical_results.get(model_name, 'N/A')
            })
        
        return final_results

if __name__ == "__main__":
    benchmark = ModelBenchmark()
    
    print(" Przygotowywanie danych testowych...")
    
    results = benchmark.benchmark_all_models()
    
    final_table = benchmark.create_comparison_table(results)
    
    print("\n" + "="*80)
    print(" COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    
    df = pd.DataFrame(final_table)
    print(df.to_string(index=False))
    
    df.to_csv('model_benchmark_results.csv', index=False)
    print(f"\n Wyniki zapisane do: model_benchmark_results.csv")
    
    print("\n Tabela do dokumentacji:")
    print("| Model | Test Accuracy (%) | Inference Time (ms) | Memory Usage (MB) | Practical Test |")
    print("|-------|------------------|---------------------|-------------------|-----------------|")
    for _, row in df.iterrows():
        print(f"| {row['Model']} | {row['Test Accuracy (%)']} | {row['Inference Time (ms)']} | {row['Memory Usage (MB)']} | {row['Practical Test']} |")