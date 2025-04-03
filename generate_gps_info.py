import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


def get_gps_info(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if exif_data is None:
            return None

        gps_info = {}
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name == 'GPSInfo':
                for gps_tag, gps_value in value.items():
                    gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                    gps_info[gps_tag_name] = gps_value
                break

        if 'GPSLatitude' in gps_info and 'GPSLatitudeRef' in gps_info and \
           'GPSLongitude' in gps_info and 'GPSLongitudeRef' in gps_info and \
           'GPSAltitude' in gps_info and 'GPSAltitudeRef' in gps_info:
            lat = convert_to_decimal(gps_info['GPSLatitude'], gps_info['GPSLatitudeRef'])
            lon = convert_to_decimal(gps_info['GPSLongitude'], gps_info['GPSLongitudeRef'])
            alt = float(gps_info['GPSAltitude']) if gps_info['GPSAltitudeRef'] == 0 else -float(gps_info['GPSAltitude'])
            return lat, lon, alt
        else:
            return None
    except Exception:
        return None


def convert_to_decimal(coordinate, ref):
    degrees = coordinate[0]
    minutes = coordinate[1]
    seconds = coordinate[2]
    decimal = float(degrees) + (float(minutes) / 60) + (float(seconds) / 3600)
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal


def scan_directory(directory, output_file):
    with open(output_file, 'w') as file:
        for root, dirs, files in os.walk(directory):
            for file_name in files:
                if file_name.lower().endswith('.jpg'):
                    image_path = os.path.join(root, file_name)
                    gps_info = get_gps_info(image_path)
                    if gps_info:
                        lat, lon, alt = gps_info
                        file.write(f"{file_name} {lat} {lon} {alt}\n")


if __name__ == "__main__":
    directory = 'control_images'  # 替换为你要扫描的目录
    output_file = 'geo_registration_info.txt'
    scan_directory(directory, output_file)
    print(f"GPS 信息已写入 {output_file}")
    