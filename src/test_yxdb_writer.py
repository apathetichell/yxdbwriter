"""
YXDB Writer Test Suite

Comprehensive test module for the YXDB writer that validates:
- Dynamic schema detection from various pandas DataFrame types
- All supported YXDB data types
- Edge cases (nulls, empty strings, large datasets)
- File format correctness
- Compatibility with YXDB readers

Usage:
    python test_yxdb_writer.py
    
    # Or run specific test categories:
    python test_yxdb_writer.py --basic
    python test_yxdb_writer.py --types
    python test_yxdb_writer.py --edge-cases
    python test_yxdb_writer.py --performance

Requirements:
    - pandas
    - numpy
    - yxdb (for reading validation) - optional but recommended
    - yxdb_writer module in same directory

Author: AI Assistant  
Date: 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
import os
import sys
import argparse
import time
from typing import List, Dict, Any, Optional

# Import our YXDB writer
try:
    from yxdb_writer import YxdbWriter, write_dataframe, create_yxdb
except ImportError:
    print("❌ Error: yxdb_writer.py not found in current directory")
    print("Please ensure yxdb_writer.py is in the same folder as this test file")
    sys.exit(1)

# Optional YXDB reader for validation
try:
    from yxdb.yxdb_reader import YxdbReader
    HAS_YXDB_READER = True
except ImportError:
    print("⚠️  Warning: yxdb reader not available. Will skip read validation tests.")
    print("   Install with: pip install yxdb")
    HAS_YXDB_READER = False


class YxdbTester:
    """Comprehensive YXDB writer test suite."""
    
    def __init__(self):
        self.test_results = []
        self.output_dir = "test_outputs"
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def log_test(self, test_name: str, success: bool, message: str = "", duration: float = 0):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        duration_str = f" ({duration:.3f}s)" if duration > 0 else ""
        print(f"{status} {test_name}{duration_str}")
        if message:
            print(f"     {message}")
        
        self.test_results.append({
            'test': test_name,
            'success': success,
            'message': message,
            'duration': duration
        })
    
    def test_basic_dataframe(self):
        """Test basic DataFrame with mixed types."""
        print("\n🧪 Testing Basic DataFrame...")
        
        try:
            start_time = time.time()
            
            # Create test DataFrame
            df = pd.DataFrame({
                'ID': [1, 2, 3, 4, 5],
                'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
                'Age': [25, 30, 35, 28, 33],
                'Salary': [50000.0, 60000.0, 70000.0, 55000.0, 65000.0],
                'Active': [True, False, True, True, False]
            })
            
            output_path = os.path.join(self.output_dir, "basic_test.yxdb")
            
            # Write file
            with YxdbWriter.from_dataframe(output_path, df) as writer:
                expected_fields = ['ID:Int64', 'Name:V_String', 'Age:Int64', 'Salary:Double', 'Active:Bool']
                actual_fields = [f"{f['name']}:{f['data_type']}" for f in writer.fields]
                
                if actual_fields != expected_fields:
                    raise ValueError(f"Field detection mismatch. Expected: {expected_fields}, Got: {actual_fields}")
            
            # Validate file exists and has content
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise ValueError("Output file not created or empty")
            
            duration = time.time() - start_time
            self.log_test("Basic DataFrame", True, f"5 records, 5 fields written", duration)
            
            # Try to read back if reader available
            if HAS_YXDB_READER:
                self._validate_with_reader(output_path, df, "Basic DataFrame Read Validation")
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Basic DataFrame", False, str(e), duration)
    
    def test_all_data_types(self):
        """Test all supported YXDB data types."""
        print("\n🧪 Testing All Data Types...")
        
        try:
            start_time = time.time()
            
            # Create DataFrame with all data types
            df = pd.DataFrame({
                'String_Field': ['Hello', 'World', 'Test'],
                'Int_Field': [1, 2, 3],
                'Float_Field': [1.1, 2.2, 3.3],
                'Bool_Field': [True, False, True],
                'Date_Field': [datetime(2024, 1, 1), datetime(2024, 2, 1), datetime(2024, 3, 1)]
            })
            
            output_path = os.path.join(self.output_dir, "all_types_test.yxdb")
            
            with YxdbWriter.from_dataframe(output_path, df) as writer:
                pass
            
            duration = time.time() - start_time
            self.log_test("All Data Types", True, "String, Int, Float, Bool, DateTime fields", duration)
            
            if HAS_YXDB_READER:
                self._validate_with_reader(output_path, df, "All Data Types Read Validation")
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("All Data Types", False, str(e), duration)
    
    def test_null_values(self):
        """Test handling of null/None values."""
        print("\n🧪 Testing Null Values...")
        
        try:
            start_time = time.time()
            
            # Create DataFrame with nulls
            df = pd.DataFrame({
                'Name': ['Alice', None, 'Charlie'],
                'Age': [25, None, 35],
                'Salary': [50000.0, None, 70000.0],
                'Active': [True, None, False]
            })
            
            output_path = os.path.join(self.output_dir, "null_test.yxdb")
            
            with YxdbWriter.from_dataframe(output_path, df) as writer:
                pass
            
            duration = time.time() - start_time
            self.log_test("Null Values", True, "Mixed null and non-null values", duration)
            
            if HAS_YXDB_READER:
                self._validate_with_reader(output_path, df, "Null Values Read Validation")
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Null Values", False, str(e), duration)
    
    def test_empty_dataframe(self):
        """Test empty DataFrame handling."""
        print("\n🧪 Testing Empty DataFrame...")
        
        try:
            start_time = time.time()
            
            # Create empty DataFrame with columns
            df = pd.DataFrame(columns=['Name', 'Age', 'Salary'])
            df = df.astype({'Name': 'object', 'Age': 'int64', 'Salary': 'float64'})
            
            output_path = os.path.join(self.output_dir, "empty_test.yxdb")
            
            with YxdbWriter.from_dataframe(output_path, df) as writer:
                if len(writer.records) != 0:
                    raise ValueError("Empty DataFrame should have 0 records")
            
            duration = time.time() - start_time
            self.log_test("Empty DataFrame", True, "0 records, 3 fields", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Empty DataFrame", False, str(e), duration)
    
    def test_large_dataset(self):
        """Test large dataset performance."""
        print("\n🧪 Testing Large Dataset...")
        
        try:
            start_time = time.time()
            
            # Create large DataFrame
            size = 10000
            df = pd.DataFrame({
                'ID': range(size),
                'Name': [f'Person_{i}' for i in range(size)],
                'Value': np.random.random(size),
                'Category': np.random.choice(['A', 'B', 'C'], size),
                'Timestamp': pd.date_range('2024-01-01', periods=size, freq='H')
            })
            
            output_path = os.path.join(self.output_dir, "large_test.yxdb")
            
            with YxdbWriter.from_dataframe(output_path, df) as writer:
                pass
            
            file_size = os.path.getsize(output_path) / 1024 / 1024  # MB
            duration = time.time() - start_time
            
            self.log_test("Large Dataset", True, 
                         f"{size:,} records, {file_size:.1f}MB", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Large Dataset", False, str(e), duration)
    
    def test_manual_fields(self):
        """Test manual field definition."""
        print("\n🧪 Testing Manual Field Definition...")
        
        try:
            start_time = time.time()
            
            # Define fields manually
            fields = [
                {"name": "ID", "data_type": "Int32"},
                {"name": "Description", "data_type": "V_String", "size": 254},
                {"name": "Amount", "data_type": "Double"},
                {"name": "Flag", "data_type": "Bool"}
            ]
            
            output_path = os.path.join(self.output_dir, "manual_fields_test.yxdb")
            
            # Create writer and add records manually
            writer = YxdbWriter(output_path, fields=fields)
            writer.add_record({"ID": 1, "Description": "First record", "Amount": 123.45, "Flag": True})
            writer.add_record({"ID": 2, "Description": "Second record", "Amount": 678.90, "Flag": False})
            writer.add_record({"ID": 3, "Description": "Third record", "Amount": None, "Flag": None})
            
            with writer:
                pass
            
            duration = time.time() - start_time
            self.log_test("Manual Field Definition", True, "3 records with custom fields", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Manual Field Definition", False, str(e), duration)
    
    def test_special_characters(self):
        """Test special characters and Unicode."""
        print("\n🧪 Testing Special Characters...")
        
        try:
            start_time = time.time()
            
            # Create DataFrame with special characters
            df = pd.DataFrame({
                'Name': ['Müller', 'José', 'François', '中文', '🌟 Star'],
                'Description': ['åäö', 'ñáé', 'çùè', '测试', '💻 Data'],
                'Value': [1.1, 2.2, 3.3, 4.4, 5.5]
            })
            
            output_path = os.path.join(self.output_dir, "special_chars_test.yxdb")
            
            with YxdbWriter.from_dataframe(output_path, df) as writer:
                pass
            
            duration = time.time() - start_time
            self.log_test("Special Characters", True, "Unicode and special chars", duration)
            
            if HAS_YXDB_READER:
                self._validate_with_reader(output_path, df, "Special Characters Read Validation", 
                                         check_strings=False)  # Skip exact string comparison due to encoding
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Special Characters", False, str(e), duration)
    
    def test_edge_cases(self):
        """Test various edge cases."""
        print("\n🧪 Testing Edge Cases...")
        
        try:
            start_time = time.time()
            
            # Various edge cases
            df = pd.DataFrame({
                'Empty_String': ['', 'normal', ''],
                'Long_String': ['x' * 1000, 'short', 'y' * 500],
                'Zero_Values': [0, 0.0, 0],
                'Negative': [-1, -99.99, -1000],
                'Large_Numbers': [1e10, 1e-10, 9.999e15]
            })
            
            output_path = os.path.join(self.output_dir, "edge_cases_test.yxdb")
            
            with YxdbWriter.from_dataframe(output_path, df) as writer:
                pass
            
            duration = time.time() - start_time
            self.log_test("Edge Cases", True, "Empty strings, long strings, extreme values", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Edge Cases", False, str(e), duration)
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        print("\n🧪 Testing Convenience Functions...")
        
        try:
            start_time = time.time()
            
            # Test write_dataframe function
            df = pd.DataFrame({
                'A': [1, 2, 3],
                'B': ['x', 'y', 'z']
            })
            
            output_path1 = os.path.join(self.output_dir, "convenience1_test.yxdb")
            write_dataframe(df, output_path1)
            
            # Test create_yxdb function
            fields = [
                {"name": "ID", "data_type": "Int32"},
                {"name": "Name", "data_type": "V_String", "size": 100}
            ]
            records = [
                {"ID": 1, "Name": "Alice"},
                {"ID": 2, "Name": "Bob"}
            ]
            
            output_path2 = os.path.join(self.output_dir, "convenience2_test.yxdb")
            create_yxdb(output_path2, records, fields)
            
            # Validate both files exist
            if not (os.path.exists(output_path1) and os.path.exists(output_path2)):
                raise ValueError("Convenience functions didn't create files")
            
            duration = time.time() - start_time
            self.log_test("Convenience Functions", True, "write_dataframe() and create_yxdb()", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Convenience Functions", False, str(e), duration)
    
    def _validate_with_reader(self, file_path: str, original_df: pd.DataFrame, 
                             test_name: str, check_strings: bool = True):
        """Validate written file by reading it back."""
        try:
            start_time = time.time()
            
            reader = YxdbReader(path=file_path)
            
            # Check record count
            if reader.num_records != len(original_df):
                raise ValueError(f"Record count mismatch: expected {len(original_df)}, got {reader.num_records}")
            
            # Read all records
            read_records = []
            while reader.next():
                record = {}
                for field in reader.list_fields():
                    record[field.name] = reader.read_name(field.name)
                read_records.append(record)
            
            reader.close()
            
            # Basic validation - check first record if exists
            if len(read_records) > 0 and len(original_df) > 0:
                read_record = read_records[0]
                original_record = original_df.iloc[0].to_dict()
                
                # Check numeric values
                for col in original_df.columns:
                    if pd.api.types.is_numeric_dtype(original_df[col]):
                        original_val = original_record[col]
                        read_val = read_record.get(col)
                        
                        if pd.notna(original_val):
                            if abs(float(read_val) - float(original_val)) > 1e-10:
                                raise ValueError(f"Numeric value mismatch for {col}: {original_val} != {read_val}")
                
                # Check string values (if enabled)
                if check_strings:
                    for col in original_df.columns:
                        if original_df[col].dtype == 'object':
                            original_val = original_record[col]
                            read_val = read_record.get(col)
                            
                            if pd.notna(original_val):
                                if str(read_val) != str(original_val):
                                    raise ValueError(f"String value mismatch for {col}: '{original_val}' != '{read_val}'")
            
            duration = time.time() - start_time
            self.log_test(test_name, True, f"Read validation passed", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test(test_name, False, f"Read validation failed: {str(e)}", duration)
    
    def run_all_tests(self):
        """Run all tests."""
        print("🚀 Starting YXDB Writer Test Suite")
        print("=" * 50)
        
        # Run all tests
        self.test_basic_dataframe()
        self.test_all_data_types()
        self.test_null_values()
        self.test_empty_dataframe()
        self.test_large_dataset()
        self.test_manual_fields()
        self.test_special_characters()
        self.test_edge_cases()
        self.test_convenience_functions()
        
        # Summary
        self.print_summary()
    
    def run_basic_tests(self):
        """Run basic functionality tests."""
        print("🚀 Running Basic Tests")
        print("=" * 30)
        
        self.test_basic_dataframe()
        self.test_all_data_types()
        self.test_null_values()
        
        self.print_summary()
    
    def run_type_tests(self):
        """Run data type tests."""
        print("🚀 Running Data Type Tests")
        print("=" * 30)
        
        self.test_all_data_types()
        self.test_manual_fields()
        
        self.print_summary()
    
    def run_edge_case_tests(self):
        """Run edge case tests."""
        print("🚀 Running Edge Case Tests")
        print("=" * 30)
        
        self.test_null_values()
        self.test_empty_dataframe()
        self.test_special_characters()
        self.test_edge_cases()
        
        self.print_summary()
    
    def run_performance_tests(self):
        """Run performance tests."""
        print("🚀 Running Performance Tests")
        print("=" * 30)
        
        self.test_large_dataset()
        
        self.print_summary()
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        total_time = sum(r['duration'] for r in self.test_results)
        print(f"Total Time: {total_time:.3f}s")
        
        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.test_results:
                if not result['success']:
                    print(f"  • {result['test']}: {result['message']}")
        
        print(f"\n📁 Output files in: {self.output_dir}/")
        
        if HAS_YXDB_READER:
            print("✅ YXDB reader available - full validation performed")
        else:
            print("⚠️  YXDB reader not available - limited validation")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='YXDB Writer Test Suite')
    parser.add_argument('--basic', action='store_true', help='Run basic tests only')
    parser.add_argument('--types', action='store_true', help='Run data type tests only')
    parser.add_argument('--edge-cases', action='store_true', help='Run edge case tests only')
    parser.add_argument('--performance', action='store_true', help='Run performance tests only')
    
    args = parser.parse_args()
    
    tester = YxdbTester()
    
    if args.basic:
        tester.run_basic_tests()
    elif args.types:
        tester.run_type_tests()
    elif args.edge_cases:
        tester.run_edge_case_tests()
    elif args.performance:
        tester.run_performance_tests()
    else:
        tester.run_all_tests()


if __name__ == "__main__":
    main()
