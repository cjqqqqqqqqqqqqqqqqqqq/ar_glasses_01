from flask import Flask, request, jsonify
from ar import process_ar_data, get_ar_system
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import webbrowser
from geopy.geocoders import GoogleV3
from geopy.distance import geodesic
import folium
import os

app = Flask(__name__)

class ARGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AR 控制面板")
        self.root.geometry("600x400")
        
        # 设置样式
        style = ttk.Style()
        style.configure('TFrame', background='#f5f5f5')
        style.configure('TButton', padding=5)
        style.configure('TLabel', background='#f5f5f5')
        
        # 主容器
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 字幕控制区域
        subtitle_frame = ttk.LabelFrame(main_frame, text="字幕控制", padding="10")
        subtitle_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(subtitle_frame, text="开启字幕", 
                  command=lambda: self.toggle_subtitle(True)).pack(side=tk.LEFT, padx=5)
        ttk.Button(subtitle_frame, text="关闭字幕", 
                  command=lambda: self.toggle_subtitle(False)).pack(side=tk.LEFT, padx=5)
        
        # 导航设置区域
        nav_frame = ttk.LabelFrame(main_frame, text="导航设置", padding="10")
        nav_frame.pack(fill=tk.X, pady=10)
        
        # 起点设置
        start_frame = ttk.Frame(nav_frame)
        start_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(start_frame, text="起点:").pack(side=tk.LEFT, padx=5)
        self.start_var = tk.StringVar()
        self.start_entry = ttk.Entry(start_frame, textvariable=self.start_var)
        self.start_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Button(start_frame, text="使用当前位置", 
                  command=self.use_current_location).pack(side=tk.LEFT, padx=5)
        
        # 目的地设置
        dest_frame = ttk.Frame(nav_frame)
        dest_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(dest_frame, text="目的地:").pack(side=tk.LEFT, padx=5)
        self.destination_var = tk.StringVar()
        self.destination_entry = ttk.Entry(dest_frame, textvariable=self.destination_var)
        self.destination_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Button(nav_frame, text="开始导航", 
                  command=self.set_navigation).pack(pady=10)
        
        # 添加 Google Maps API key
        self.geolocator = GoogleV3(api_key='你的API密钥')  # 替换这里
        
        # 添加地图按钮
        ttk.Button(nav_frame, text="在地图中查看", 
                  command=self.show_map).pack(pady=5)
        
        # 添加危险速度阈值设置区域
        speed_frame = ttk.LabelFrame(main_frame, text="危险速度阈值设置", padding="10")
        speed_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(speed_frame, text="阈值 (km/h):").pack(side=tk.LEFT, padx=5)
        self.speed_var = tk.StringVar(value="72")  # 默认值 20m/s ≈ 72km/h
        self.speed_entry = ttk.Entry(speed_frame, textvariable=self.speed_var, width=10)
        self.speed_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(speed_frame, text="更新阈值", 
                  command=self.update_speed_threshold).pack(side=tk.LEFT, padx=5)
        
        # 状态显示区域
        self.status_label = ttk.Label(main_frame, text="就绪", wraplength=550)
        self.status_label.pack(fill=tk.X, pady=10)

    def toggle_subtitle(self, enabled):
        try:
            ar_system = get_ar_system()  # 获取AR系统实例
            ar_system.toggle_subtitle(enabled)  # 使用新的切换方法
            status = "开启" if enabled else "关闭"
            self.status_label.config(text=f"字幕已{status}")
            messagebox.showinfo("成功", f"字幕已{status}")
        except Exception as e:
            messagebox.showerror("错误", str(e))
            
    def use_current_location(self):
        """使用当前位置作为起点"""
        try:
            current_location = process_ar_data({'get_current_location': True})
            if current_location.get('location'):
                self.start_var.set(current_location['location'])
                self.status_label.config(text="已设置当前位置为起点")
            else:
                messagebox.showerror("错误", "无法获取当前位置")
        except Exception as e:
            messagebox.showerror("错误", str(e))
            
    def set_navigation(self):
        destination = self.destination_var.get().strip()
        start_location = self.start_var.get().strip()
        
        if not destination:
            messagebox.showwarning("警告", "请输入目的地")
            return
            
        try:
            # 地理编码：将地址转换为坐标
            try:
                dest_loc = self.geolocator.geocode(destination)
                if not dest_loc:
                    raise ValueError("无法找到目的地位置")
                dest_coords = (dest_loc.latitude, dest_loc.longitude)
                
                if start_location:
                    start_loc = self.geolocator.geocode(start_location)
                    if not start_loc:
                        raise ValueError("无法找到起点位置")
                    start_coords = (start_loc.latitude, start_loc.longitude)
                else:
                    ar_system = get_ar_system()
                    start_coords = ar_system.current_position
                
                # 更新 AR 系统
                ar_system = get_ar_system()
                ar_system.current_position = start_coords
                ar_system.target_position = dest_coords
                ar_system.navigation_active = True
                
                # 计算距离
                distance = geodesic(start_coords, dest_coords).kilometers
                
                self.status_label.config(
                    text=f"导航已设置\n"
                         f"从: {start_location or '当前位置'}\n"
                         f"到: {destination}\n"
                         f"距离: {distance:.2f} 公里"
                )
                messagebox.showinfo("成功", "导航路线已设置")
                
            except ValueError as ve:
                messagebox.showerror("错误", str(ve))
            except Exception as e:
                messagebox.showerror("错误", f"地理编码错误: {str(e)}")
                
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def update_speed_threshold(self):
        """更新危险速度阈值"""
        try:
            speed_kmh = float(self.speed_var.get())
            if speed_kmh < 0:
                messagebox.showwarning("警告", "速度阈值不能为负数")
                return
                
            # 转换为 m/s
            speed_ms = speed_kmh * 1000 / 3600
            
            ar_system = get_ar_system()  # 获取AR系统实例
            ar_system.danger_speed_threshold = speed_ms  # 直接设置速度阈值
            
            self.status_label.config(text=f"危险速度阈值已更新为: {speed_kmh} km/h")
            messagebox.showinfo("成功", f"速度阈值已更新为 {speed_kmh} km/h")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def show_map(self):
        """在浏览器中显示地图"""
        try:
            ar_system = get_ar_system()
            if not ar_system.target_position:
                messagebox.showwarning("警告", "请先设置导航路线")
                return
                
            # 创建地图
            center_lat = (ar_system.current_position[0] + ar_system.target_position[0]) / 2
            center_lon = (ar_system.current_position[1] + ar_system.target_position[1]) / 2
            m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
            
            # 添加起点标记
            folium.Marker(
                ar_system.current_position,
                popup='起点',
                icon=folium.Icon(color='green')
            ).add_to(m)
            
            # 添加终点标记
            folium.Marker(
                ar_system.target_position,
                popup='终点',
                icon=folium.Icon(color='red')
            ).add_to(m)
            
            # 添加路线
            folium.PolyLine(
                locations=[ar_system.current_position, ar_system.target_position],
                weight=2,
                color='blue',
                opacity=0.8
            ).add_to(m)
            
            # 保存并显示地图
            map_path = os.path.join(os.path.dirname(__file__), 'navigation_map.html')
            m.save(map_path)
            webbrowser.open('file://' + os.path.abspath(map_path))
            
        except Exception as e:
            messagebox.showerror("错误", f"无法显示地图: {str(e)}")

@app.route('/navigation/settings', methods=['GET', 'POST'])
def navigation_settings():
    try:
        if request.method == 'POST':
            data = request.get_json()
            
            # 检查必要参数
            if 'destination' not in data:
                return jsonify({'error': '缺少必要参数：destination'}), 400
            
            # 准备导航设置
            nav_settings = {
                'destination': data['destination'],
                'update_navigation': True
            }
            
            # 如果提供了起点，使用提供的起点；否则使用当前位置
            if 'start_location' in data:
                nav_settings['start_location'] = data['start_location']
            else:
                # 获取当前位置
                current_location = process_ar_data({'get_current_location': True})
                if not current_location.get('location'):
                    return jsonify({'error': '无法获取当前位置'}), 400
                nav_settings['start_location'] = current_location['location']
            
            result = process_ar_data(nav_settings)
            return jsonify({
                'status': 'success',
                'navigation': {
                    'start_location': nav_settings['start_location'],
                    'destination': data['destination']
                }
            })
            
        else:  # GET 请求
            # 获取当前导航设置
            current_settings = process_ar_data({'get_navigation_settings': True})
            return jsonify(current_settings)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_flask():
    app.run(host='0.0.0.0', port=5000)

def main():
    # 启动 AR 系统在单独的线程中
    ar_system = get_ar_system()
    ar_thread = threading.Thread(target=ar_system.run, daemon=True)
    ar_thread.start()
    
    # 启动 Flask 服务器在单独的线程中
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # 创建 GUI 窗口
    root = tk.Tk()
    gui = ARGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
