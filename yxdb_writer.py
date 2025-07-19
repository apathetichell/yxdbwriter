"""
YXDB Writer - Standalone module for writing Alteryx YXDB files

This module provides a complete implementation for writing YXDB files from pandas DataFrames
or manual record addition. It automatically detects field types from DataFrame dtypes and 
handles all the complex YXDB format requirements including LZF compression and blob encoding.

Usage:
    # From pandas DataFrame (automatic schema detection)
    import pandas as pd
    from yxdb_writer import YxdbWriter
    
    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'Salary': [50000.0, 60000.0, 70000.0],
        'Active': [True, False, True]
    })
    
    with YxdbWriter.from_dataframe("output.yxdb", df) as writer:
        pass  # File written automatically
    
    # Manual field definition and record addition
    fields = [
        {"name": "ID", "data_type": "Int32"},
        {"name": "Description", "data_type": "V_String", "size": 254},
        {"name": "Value", "data_type": "Double"}
    ]
    
    writer = YxdbWriter("manual.yxdb", fields=fields)
    writer.add_record({"ID": 1, "Description": "First record", "Value": 123.45})
    writer.add_record({"ID": 2, "Description": "Second record", "Value": 678.90})
    
    with writer:
        pass  # Write to file

Author: AI Assistant
Date: 2025
License: MIT
"""

import struct
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, date
from io import BytesIO
import xml.etree.ElementTree as ET


def create_uncompressed_lzf_block(data: bytes) -> bytes:
    """
    Create an LZF block that stores data uncompressed but in valid LZF format.
    
    YXDB files require LZF format even for uncompressed data. This function
    creates proper LZF literal blocks that the YXDB reader can decompress.
    
    Args:
        data: Raw byte data to wrap in LZF format
        
    Returns:
        LZF-formatted byte data
    """
    result = BytesIO()
    pos = 0
    
    while pos < len(data):
        chunk_size = min(32, len(data) - pos)
        control_byte = chunk_size - 1  # 0-31 for literals (length - 1)
        
        result.write(bytes([control_byte]))
        result.write(data[pos:pos + chunk_size])
        pos += chunk_size
    
    return result.getvalue()


class YxdbWriter:
    """
    YXDB Writer for creating Alteryx-compatible database files.
    
    This class handles the complete YXDB file format including:
    - Header and metadata sections
    - All standard YXDB data types
    - Variable-length string encoding (V_String blob format)
    - Proper null flag positioning
    - LZF compression
    - Multiple record serialization
    
    Attributes:
        path (str): Output file path
        fields (List[Dict]): Field definitions with name, data_type, size, scale
        records (List[Dict]): Records to be written
    """
    
    def __init__(self, path: str, dataframe: Optional[pd.DataFrame] = None, 
                 fields: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize YXDB writer.
        
        Args:
            path: Output file path
            dataframe: Optional pandas DataFrame to write
            fields: Optional field definitions. If not provided with DataFrame,
                   fields will be auto-detected from DataFrame dtypes
        """
        self.path = path
        self.dataframe = dataframe
        self.fields = []
        self.records = []
        self.file = None
        
        if dataframe is not None:
            if fields is None:
                self.fields = self._detect_fields_from_dataframe(dataframe)
            else:
                self.fields = self._normalize_fields(fields)
            
            self.records = self._dataframe_to_records(dataframe)
        elif fields is not None:
            self.fields = self._normalize_fields(fields)
    
    def _detect_fields_from_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Automatically detect YXDB field types from pandas DataFrame dtypes.
        
        Maps pandas dtypes to appropriate YXDB data types:
        - float -> Double
        - int -> Int64  
        - bool -> Bool
        - datetime -> DateTime
        - object/string -> V_String (variable-length string)
        
        Args:
            df: pandas DataFrame
            
        Returns:
            List of field definitions
        """
        fields = []
        
        for col_name in df.columns:
            dtype = df[col_name].dtype
            field_def = {'name': str(col_name)}
            
            if pd.api.types.is_float_dtype(dtype):
                field_def.update({'data_type': 'Double', 'size': 0, 'scale': 0})
            elif pd.api.types.is_integer_dtype(dtype):
                field_def.update({'data_type': 'Int64', 'size': 0, 'scale': 0})
            elif pd.api.types.is_bool_dtype(dtype):
                field_def.update({'data_type': 'Bool', 'size': 0, 'scale': 0})
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                field_def.update({'data_type': 'DateTime', 'size': 0, 'scale': 0})
            else:
                # Default to V_String for all other types (strings, objects, etc.)
                field_def.update({'data_type': 'V_String', 'size': 254, 'scale': 0})
            
            fields.append(field_def)
        
        return fields
    
    def _normalize_fields(self, fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize field definitions to standard format.
        
        Ensures all field definitions have consistent keys and data types.
        
        Args:
            fields: List of field definitions
            
        Returns:
            Normalized field definitions
        """
        normalized = []
        for field in fields:
            normalized_field = {
                'name': field['name'],
                'data_type': field.get('type', field.get('data_type')),
                'size': field.get('size', 0),
                'scale': field.get('scale', 0)
            }
            normalized.append(normalized_field)
        return normalized
    
    def _dataframe_to_records(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert pandas DataFrame to list of record dictionaries.
        
        Handles pandas-specific types and converts them to Python natives:
        - numpy scalars -> Python scalars
        - pandas Timestamps -> Python datetime
        - NaN values -> None
        
        Args:
            df: pandas DataFrame
            
        Returns:
            List of record dictionaries
        """
        records = []
        for _, row in df.iterrows():
            record = {}
            for col_name in df.columns:
                value = row[col_name]
                if pd.isna(value):
                    record[col_name] = None
                else:
                    if isinstance(value, (np.integer, np.floating, np.bool_)):
                        record[col_name] = value.item()
                    elif isinstance(value, pd.Timestamp):
                        record[col_name] = value.to_pydatetime()
                    else:
                        record[col_name] = value
            records.append(record)
        return records
    
    @classmethod
    def from_dataframe(cls, path: str, df: pd.DataFrame, 
                      fields: Optional[List[Dict[str, Any]]] = None) -> 'YxdbWriter':
        """
        Create YxdbWriter from pandas DataFrame.
        
        Convenience method for the most common use case.
        
        Args:
            path: Output file path
            df: pandas DataFrame
            fields: Optional field definitions (auto-detected if not provided)
            
        Returns:
            YxdbWriter instance
        """
        return cls(path, dataframe=df, fields=fields)
    
    def __enter__(self):
        """Context manager entry - opens file for writing."""
        self.file = open(self.path, 'wb')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - writes file and closes."""
        if self.file:
            self._write_file()
            self.file.close()
    
    def add_record(self, record: Dict[str, Any]):
        """
        Add a single record to be written.
        
        Args:
            record: Dictionary with field names as keys and values
        """
        self.records.append(record)
    
    def _write_file(self):
        """Write complete YXDB file (header + metadata + data)."""
        self._write_header()
        self._write_metadata()
        self._write_data()
    
    def _write_header(self):
        """Write 512-byte YXDB header with signature and record count."""
        header = bytearray(512)
        
        # YXDB signature
        header[0:21] = b"Alteryx Database File"
        
        # Store position for metadata size (written later)
        self.meta_size_pos = 80
        
        # Number of records at position 104
        struct.pack_into('<I', header, 104, len(self.records))
        
        self.file.write(header)
    
    def _write_metadata(self):
        """Write XML metadata section describing fields."""
        xml_str = self._generate_xml_metadata()
        xml_bytes = xml_str.encode('utf-16le')
        meta_size = len(xml_bytes) // 2 + 1
        
        # Update header with metadata size
        current_pos = self.file.tell()
        self.file.seek(self.meta_size_pos)
        self.file.write(struct.pack('<I', meta_size))
        self.file.seek(current_pos)
        
        # Write metadata and terminator
        self.file.write(xml_bytes)
        self.file.write(b'\x00\x00')
    
    def _generate_xml_metadata(self) -> str:
        """
        Generate XML metadata describing the record structure.
        
        Creates proper YXDB metadata with MetaInfo wrapper and field definitions.
        
        Returns:
            XML string
        """
        meta_info = ET.Element("MetaInfo")
        meta_info.set("connection", "Output")
        record_info = ET.SubElement(meta_info, "RecordInfo")
        
        for field in self.fields:
            field_elem = ET.SubElement(record_info, "Field")
            field_elem.set("name", field['name'])
            field_elem.set("type", field['data_type'])
            if field['size'] > 0:
                field_elem.set("size", str(field['size']))
            if field['scale'] > 0:
                field_elem.set("scale", str(field['scale']))
        
        self._indent_xml(meta_info)
        return ET.tostring(meta_info, encoding='unicode')
    
    def _indent_xml(self, elem, level=0):
        """Add proper indentation to XML elements."""
        i = "\n" + level * "\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                self._indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
    
    def _write_data(self):
        """Write all records in LZF-compressed format."""
        if not self.records:
            return
        
        # Serialize all records
        all_records_data = BytesIO()
        for record in self.records:
            record_data = self._serialize_record(record)
            all_records_data.write(record_data)
        
        # Create LZF block
        raw_data = all_records_data.getvalue()
        lzf_data = create_uncompressed_lzf_block(raw_data)
        block_length = len(lzf_data)
        
        # Write block length and data
        self.file.write(struct.pack('<I', block_length))
        self.file.write(lzf_data)
    
    def _serialize_record(self, record: Dict[str, Any]) -> bytes:
        """
        Serialize a single record using the correct YXDB binary format.
        
        This is the core serialization logic that handles:
        - Fixed vs variable field positioning
        - Null flag placement (varies by data type)
        - V_String blob encoding
        - Proper byte alignment
        
        Args:
            record: Dictionary of field values
            
        Returns:
            Serialized record bytes
        """
        fixed_size = self._calculate_fixed_size()
        has_var = self._has_variable_fields()
        
        # Create variable data first (for V_String, Blob fields)
        var_data = BytesIO()
        var_offsets = {}
        
        for field in self.fields:
            field_name = field['name']
            field_type = field['data_type']
            value = record.get(field_name)
            
            if field_type in ['V_String', 'V_WString']:
                if value is None or value == '':
                    var_offsets[field_name] = None
                else:
                    # Store RELATIVE offset from the field position to the variable data
                    # The reader calculates: block_start = field_pos + offset
                    # We want block_start to point to our variable data
                    field_pos_in_buffer = 0  # Will be calculated later when we know position
                    var_data_pos = fixed_size + 4 + var_data.tell()
                    
                    # For now, store the var_data position, we'll calculate relative offset later
                    var_offsets[field_name] = var_data.tell()
                    
                    # Encode string with proper encoding and error handling
                    if field_type == 'V_String':
                        # Try latin-1 first, fall back to utf-8 if needed
                        try:
                            str_bytes = str(value).encode('latin1')
                        except UnicodeEncodeError:
                            # For Unicode characters, encode as UTF-8 but truncate if too long
                            str_bytes = str(value).encode('utf-8')
                    else:  # V_WString
                        str_bytes = str(value).encode('utf-16le')
                    
                    # Limit string length to prevent issues
                    max_length = 255  # Reasonable limit for small blob format
                    if len(str_bytes) > max_length:
                        str_bytes = str_bytes[:max_length]
                    
                    # Write as small blob: header byte + data
                    string_length = len(str_bytes)
                    if string_length > 127:
                        # Use normal blob format for longer strings
                        # Length as 4 bytes, then data
                        var_data.write(struct.pack('<I', string_length * 2))  # *2 for normal blob format
                        var_data.write(str_bytes)
                    else:
                        # Use small blob format
                        small_block_header = (string_length << 1) | 1  # Length in high 7 bits, bit 0 = 1
                        var_data.write(bytes([small_block_header]))
                        var_data.write(str_bytes)
            
            elif field_type in ['Blob', 'SpatialObj']:
                if value is None:
                    var_offsets[field_name] = None
                else:
                    var_offsets[field_name] = fixed_size + 4 + var_data.tell()
                    if isinstance(value, str):
                        blob_bytes = value.encode('utf-8')
                    else:
                        blob_bytes = bytes(value)
                    
                    # Simple blob format (could be enhanced for complex blobs)
                    blob_length = len(blob_bytes)
                    var_data.write(struct.pack('<I', blob_length))
                    var_data.write(blob_bytes)
        
        # Create complete record buffer
        var_data_bytes = var_data.getvalue()
        var_length = len(var_data_bytes)
        total_size = fixed_size + (4 if has_var else 0) + var_length
        
        record_buffer = bytearray(total_size)
        
        # Write fixed data with correct field spacing and null flag positioning
        pos = 0
        for field in self.fields:
            field_name = field['name']
            field_type = field['data_type']
            value = record.get(field_name)
            
            if field_type in ['V_String', 'V_WString', 'Blob', 'SpatialObj']:
                # 4 bytes: offset or special value
                if var_offsets[field_name] is None:
                    if value is None:
                        struct.pack_into('<I', record_buffer, pos, 1)  # NULL marker
                    else:
                        struct.pack_into('<I', record_buffer, pos, 0)  # Empty string marker
                else:
                    # Calculate relative offset from this field position to the variable data
                    # The reader calculates: block_start = field_pos + offset
                    # We want: field_pos + offset = var_data_start + var_offset
                    # So: offset = (var_data_start + var_offset) - field_pos
                    var_data_start = fixed_size + 4  # Where variable data begins in buffer
                    var_offset = var_offsets[field_name]  # Offset within variable data
                    relative_offset = (var_data_start + var_offset) - pos
                    
                    struct.pack_into('<I', record_buffer, pos, relative_offset)
                pos += 4
            
            elif field_type == 'Double':
                # 8 bytes data + 1 byte null flag AT END
                if value is None:
                    record_buffer[pos + 8] = 1  # Null flag at end
                else:
                    struct.pack_into('<d', record_buffer, pos, float(value))
                    record_buffer[pos + 8] = 0  # Not null
                pos += 9
            
            elif field_type == 'Int64':
                # 8 bytes data + 1 byte null flag AT END
                if value is None:
                    record_buffer[pos + 8] = 1
                else:
                    struct.pack_into('<q', record_buffer, pos, int(value))
                    record_buffer[pos + 8] = 0
                pos += 9
            
            elif field_type == 'Int32':
                # 4 bytes data + 1 byte null flag AT END
                if value is None:
                    record_buffer[pos + 4] = 1
                else:
                    struct.pack_into('<i', record_buffer, pos, int(value))
                    record_buffer[pos + 4] = 0
                pos += 5
            
            elif field_type == 'Int16':
                # 2 bytes data + 1 byte null flag AT END
                if value is None:
                    record_buffer[pos + 2] = 1
                else:
                    struct.pack_into('<h', record_buffer, pos, int(value))
                    record_buffer[pos + 2] = 0
                pos += 3
            
            elif field_type == 'Float':
                # 4 bytes data + 1 byte null flag AT END
                if value is None:
                    record_buffer[pos + 4] = 1
                else:
                    struct.pack_into('<f', record_buffer, pos, float(value))
                    record_buffer[pos + 4] = 0
                pos += 5
            
            elif field_type == 'Bool':
                # 1 byte: 0=false, 1=true, 2=null (NO separate null flag)
                if value is None:
                    record_buffer[pos] = 2
                else:
                    record_buffer[pos] = 1 if value else 0
                pos += 1
            
            elif field_type == 'Byte':
                # 1 byte data + 1 byte null flag AT END
                if value is None:
                    record_buffer[pos + 1] = 1
                else:
                    # Ensure byte value is in valid range
                    byte_val = int(value) & 0xFF
                    record_buffer[pos] = byte_val
                    record_buffer[pos + 1] = 0
                pos += 2
            
            elif field_type == 'Date':
                # 10 bytes data + 1 byte null flag AT END
                if value is None:
                    record_buffer[pos + 10] = 1
                else:
                    if isinstance(value, str):
                        dt = datetime.strptime(value, '%Y-%m-%d')
                    elif isinstance(value, date):
                        dt = datetime.combine(value, datetime.min.time())
                    else:
                        dt = value
                    
                    date_str = dt.strftime('%Y-%m-%d')
                    date_bytes = date_str.encode('ascii')
                    record_buffer[pos:pos+len(date_bytes)] = date_bytes
                    record_buffer[pos + 10] = 0
                pos += 11
            
            elif field_type == 'DateTime':
                # 19 bytes data + 1 byte null flag AT END
                if value is None:
                    record_buffer[pos + 19] = 1
                else:
                    if isinstance(value, str):
                        dt = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                    else:
                        dt = value
                    
                    dt_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                    dt_bytes = dt_str.encode('ascii')
                    record_buffer[pos:pos+len(dt_bytes)] = dt_bytes
                    record_buffer[pos + 19] = 0
                pos += 20
            
            elif field_type == 'String':
                # Fixed-length string: size bytes + 1 null flag AT END
                size = field['size']
                if value is None:
                    record_buffer[pos + size] = 1
                else:
                    str_bytes = str(value).encode('utf-8')
                    if len(str_bytes) > size:
                        str_bytes = str_bytes[:size]
                    record_buffer[pos:pos+len(str_bytes)] = str_bytes
                    record_buffer[pos + size] = 0
                pos += size + 1
            
            elif field_type == 'WString':
                # Fixed-length wide string: (size * 2) bytes + 1 null flag AT END
                size = field['size']
                if value is None:
                    record_buffer[pos + (size * 2)] = 1
                else:
                    str_bytes = str(value).encode('utf-16le')
                    if len(str_bytes) > size * 2:
                        str_bytes = str_bytes[:size * 2]
                    record_buffer[pos:pos+len(str_bytes)] = str_bytes
                    record_buffer[pos + (size * 2)] = 0
                pos += (size * 2) + 1
            
            elif field_type == 'FixedDecimal':
                # Fixed decimal: size bytes + 1 null flag AT END
                size = field['size']
                scale = field['scale']
                if value is None:
                    record_buffer[pos + size] = 1
                else:
                    # Convert to scaled integer
                    scaled = int(float(value) * (10 ** scale))
                    scaled_bytes = struct.pack('<q', scaled)[:size]
                    record_buffer[pos:pos+size] = scaled_bytes
                    record_buffer[pos + size] = 0
                pos += size + 1
        
        # Add variable data length and data
        if has_var:
            struct.pack_into('<I', record_buffer, fixed_size, var_length)
            if var_length > 0:
                record_buffer[fixed_size + 4:fixed_size + 4 + var_length] = var_data_bytes
        
        return bytes(record_buffer)
    
    def _calculate_fixed_size(self) -> int:
        """
        Calculate the fixed portion size of each record.
        
        This determines how much space each record needs for non-variable fields.
        
        Returns:
            Fixed size in bytes
        """
        size = 0
        for field in self.fields:
            field_type = field['data_type']
            
            if field_type in ['V_String', 'V_WString', 'Blob', 'SpatialObj']:
                size += 4  # 4-byte offset
            elif field_type in ['Double', 'Int64']:
                size += 9  # 8 bytes data + 1 null flag
            elif field_type in ['Int32', 'Float']:
                size += 5  # 4 bytes data + 1 null flag
            elif field_type == 'Int16':
                size += 3  # 2 bytes data + 1 null flag
            elif field_type == 'Bool':
                size += 1  # 1 byte (includes null flag)
            elif field_type == 'Byte':
                size += 2  # 1 byte data + 1 null flag
            elif field_type == 'Date':
                size += 11  # 10 bytes data + 1 null flag
            elif field_type == 'DateTime':
                size += 20  # 19 bytes data + 1 null flag
            elif field_type == 'String':
                size += field['size'] + 1  # size bytes + 1 null flag
            elif field_type == 'WString':
                size += (field['size'] * 2) + 1  # (size * 2) bytes + 1 null flag
            elif field_type == 'FixedDecimal':
                size += field['size'] + 1  # size bytes + 1 null flag
        
        return size
    
    def _has_variable_fields(self) -> bool:
        """
        Check if any fields require variable-length storage.
        
        Returns:
            True if any variable fields exist
        """
        for field in self.fields:
            if field['data_type'] in ['V_String', 'V_WString', 'Blob', 'SpatialObj']:
                return True
        return False


# Convenience functions for common use cases
def write_dataframe(df: pd.DataFrame, path: str, 
                   fields: Optional[List[Dict[str, Any]]] = None) -> None:
    """
    Write pandas DataFrame to YXDB file.
    
    Simple convenience function for the most common use case.
    
    Args:
        df: pandas DataFrame to write
        path: Output file path
        fields: Optional field definitions (auto-detected if not provided)
    """
    with YxdbWriter.from_dataframe(path, df, fields) as writer:
        pass


def create_yxdb(path: str, records: List[Dict[str, Any]], 
               fields: List[Dict[str, Any]]) -> None:
    """
    Create YXDB file from records and field definitions.
    
    Args:
        path: Output file path
        records: List of record dictionaries
        fields: Field definitions
    """
    writer = YxdbWriter(path, fields=fields)
    for record in records:
        writer.add_record(record)
    
    with writer:
        pass


if __name__ == "__main__":
    # Example usage
    print("YXDB Writer - Example Usage")
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'ID': [1, 2, 3],
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'Salary': [50000.0, 60000.0, 70000.0],
        'Active': [True, False, True]
    })
    
    print("Writing DataFrame to example.yxdb...")
    with YxdbWriter.from_dataframe("example.yxdb", df) as writer:
        print(f"Fields detected: {[f['name'] + ':' + f['data_type'] for f in writer.fields]}")
        print(f"Records to write: {len(writer.records)}")
    
    print("✅ Example file created successfully!")