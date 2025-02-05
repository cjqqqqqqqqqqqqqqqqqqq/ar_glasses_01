import cv2
import numpy as np
from cv2 import aruco
import speech_recognition as sr
import threading
import queue
import time
import math
import random
import serial  # 用于GPS模块通信
from geopy import distance  # 用于计算GPS距离
from geopy.point import Point
from heapq import heappush, heappop
from scipy.special import comb  # 用于贝塞尔曲线计算
from scipy import interpolate  # 用于路径平滑
from PIL import ImageFont, ImageDraw, Image
from geopy.distance import geodesic

class PathPlanner:
    def __init__(self):
        self.obstacles = []  # 存储障碍物位置
        self.grid_size = 20  # 网格大小
        self.smooth_factor = 0.5  # 路径平滑因子
        self.safety_distance = 15  # 安全距离
        self.path_resolution = 10  # 路径分辨率
        
    def add_obstacle(self, pos):
        """添加障碍物"""
        self.obstacles.append(pos)
        
    def is_valid(self, pos, frame_shape):
        """检查位置是否有效"""
        x, y = pos
        height, width = frame_shape[:2]
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
        # 检查是否与障碍物碰撞
        for obs in self.obstacles:
            if distance.distance(pos, obs).meters < 10:  # 10米安全距离
                return False
        return True
        
    def get_neighbors(self, pos, frame_shape):
        """获取相邻的有效位置"""
        x, y = pos
        neighbors = []
        directions = [(0,1), (1,0), (0,-1), (-1,0), 
                     (1,1), (1,-1), (-1,1), (-1,-1)]  # 8个方向
        
        for dx, dy in directions:
            new_pos = (x + dx * self.grid_size, y + dy * self.grid_size)
            if self.is_valid(new_pos, frame_shape):
                neighbors.append(new_pos)
        return neighbors
        
    def heuristic(self, pos1, pos2):
        """估计两点间距离"""
        return distance.distance(pos1, pos2).meters
        
    def bezier_curve(self, points, num_points=100):
        """使用贝塞尔曲线平滑路径"""
        n = len(points) - 1
        t = np.linspace(0, 1, num_points)
        path = np.zeros((num_points, 2))
        
        for i in range(n + 1):
            coef = comb(n, i) * (1 - t)**(n - i) * t**i
            path += np.outer(coef, points[i])
            
        return path
        
    def smooth_path(self, path):
        """平滑路径"""
        if len(path) < 3:
            return path
            
        # 使用B样条插值
        x = [p[0] for p in path]
        y = [p[1] for p in path]
        
        # 创建参数化的点
        t = np.linspace(0, 1, len(path))
        
        # 使用B样条插值
        cs_x = interpolate.CubicSpline(t, x)
        cs_y = interpolate.CubicSpline(t, y)
        
        # 生成更密集的点
        t_new = np.linspace(0, 1, len(path) * 5)
        smooth_path = list(zip(cs_x(t_new), cs_y(t_new)))
        
        return self.optimize_smooth_path(smooth_path)
        
    def optimize_smooth_path(self, path):
        """优化平滑后的路径，确保安全距离"""
        optimized_path = []
        for point in path:
            if self.is_safe_point(point):
                optimized_path.append(point)
        return optimized_path
        
    def is_safe_point(self, point):
        """检查点是否满足安全距离要求"""
        for obs in self.obstacles:
            if distance.distance(point, obs).meters < self.safety_distance:
                return False
        return True
        
    def calculate_path_cost(self, path):
        """计算路径代价"""
        if not path:
            return float('inf')
            
        cost = 0
        # 计算路径长度
        for i in range(len(path)-1):
            cost += distance.distance(path[i], path[i+1]).meters
            
        # 考虑转弯次数
        for i in range(1, len(path)-1):
            v1 = np.array(path[i]) - np.array(path[i-1])
            v2 = np.array(path[i+1]) - np.array(path[i])
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            cost += angle * 10  # 转弯惩罚
            
        return cost
        
    def plan_path(self, start, goal, frame_shape):
        """优化后的A*路径规划"""
        # 使用启发式搜索的A*算法
        open_set = []
        heappush(open_set, (0, start))
        came_from = {start: None}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current = heappop(open_set)[1]
            
            if self.is_goal_reached(current, goal):
                path = self.reconstruct_path(came_from, current)
                # 平滑和优化路径
                smooth_path = self.smooth_path(path)
                return self.post_process_path(smooth_path)
                
            for neighbor in self.get_neighbors(current, frame_shape):
                tentative_g_score = g_score[current] + self.movement_cost(current, neighbor)
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        return []
        
    def movement_cost(self, current, next_pos):
        """计算移动代价"""
        base_cost = distance.distance(current, next_pos).meters
        
        # 考虑与障碍物的距离
        min_obstacle_dist = float('inf')
        for obs in self.obstacles:
            dist = distance.distance(next_pos, obs).meters
            min_obstacle_dist = min(min_obstacle_dist, dist)
        
        # 距离障碍物越近，代价越高
        obstacle_cost = max(0, self.safety_distance - min_obstacle_dist) * 2
        
        return base_cost + obstacle_cost
        
    def is_goal_reached(self, current, goal):
        """检查是否到达目标"""
        return distance.distance(current, goal).meters < self.path_resolution
        
    def post_process_path(self, path):
        """路径后处理"""
        if not path:
            return path
            
        # 移除冗余点
        simplified_path = [path[0]]
        for i in range(1, len(path)-1):
            prev_dir = np.array(path[i]) - np.array(path[i-1])
            next_dir = np.array(path[i+1]) - np.array(path[i])
            if not np.allclose(prev_dir/np.linalg.norm(prev_dir), 
                             next_dir/np.linalg.norm(next_dir)):
                simplified_path.append(path[i])
        simplified_path.append(path[-1])
        
        return simplified_path

class ARSystem:
    def __init__(self):
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        
        # 设置ArUco标记参数 - 使用正确的Dictionary创建方式
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # 设置相机参数（这些参数需要通过相机标定获得，这里使用示例值）
        self.camera_matrix = np.array([[800, 0, 320],
                                     [0, 800, 240],
                                     [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((4,1))
        
        # 添加字幕相关的属性
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.subtitle_text = ""
        self.subtitle_queue = queue.Queue()
        self.last_subtitle_time = 0
        self.subtitle_duration = 5
        self.subtitle_enabled = True  # 添加字幕开关状态
        
        # 添加雷达和导航相关的属性
        self.radar_angle = 0  # 雷达扫描角度
        self.radar_points = []  # 雷达检测到的点
        self.target_position = None  # 目标位置
        self.navigation_active = False
        
        # GPS相关属性
        try:
            self.gps_serial = serial.Serial('/dev/ttyUSB0', 9600)  # GPS串口配置
        except:
            print("警告：无法连接GPS模块")
            self.gps_serial = None
            
        self.current_position = (39.9042, 116.4074)  # 默认位置
        self.target_position = None   # 目标GPS位置
        
        # 运动检测相关属性
        self.prev_frame = None
        self.motion_points = []
        self.min_motion_area = 1000  # 增加最小运动检测面积
        self.last_motion_time = {}  # 用于跟踪每个运动点的最后更新时间
        self.motion_timeout = 0.5  # 运动点的超时时间（秒）
        
        # 添加路径规划相关的属性
        self.path_planner = PathPlanner()
        self.current_path = []
        self.path_color = (0, 255, 255)  # 黄色路径
        self.obstacle_color = (0, 0, 255)  # 红色障碍物
        
        # 添加速度阈值
        self.danger_speed_threshold = 30  # 危险速度阈值
        
        # 启动语音识别线程
        self.speech_thread = threading.Thread(target=self.speech_recognition_thread)
        self.speech_thread.daemon = True
        self.speech_thread.start()
        
        # 启动雷达扫描线程
        self.radar_thread = threading.Thread(target=self.radar_scan_thread)
        self.radar_thread.daemon = True
        self.radar_thread.start()
        
        # 启动GPS更新线程
        self.gps_thread = threading.Thread(target=self.gps_update_thread)
        self.gps_thread.daemon = True
        self.gps_thread.start()

    def speech_recognition_thread(self):
        """持续进行语音识别的线程"""
        print("正在初始化语音识别...")
        
        # 列出可用的麦克风
        print("可用的麦克风设备：")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"麦克风 {index}: {name}")
        
        try:
            # 使用默认麦克风
            self.microphone = sr.Microphone()
            print(f"使用默认麦克风")
            
            with self.microphone as source:
                # 调整环境噪声
                print("正在调整环境噪声...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                print("语音识别准备就绪")
            
            while True:
                try:
                    with self.microphone as source:
                        print("正在监听...")
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                        
                    try:
                        print("正在识别...")
                        # 尝试使用本地语音识别（如果可用）
                        try:
                            text = self.recognizer.recognize_sphinx(audio, language='zh-cn')
                        except:
                            # 如果本地识别失败，使用Google在线服务
                            text = self.recognizer.recognize_google(audio, language='zh-CN')
                        
                        print(f"识别结果: {text}")
                        if text:  # 只有在有实际文本时才更新字幕
                            self.subtitle_queue.put(text)
                            
                    except sr.UnknownValueError:
                        print("无法识别语音")
                    except sr.RequestError as e:
                        error_msg = f"语音识别服务错误: {str(e)}"
                        print(error_msg)
                        self.subtitle_queue.put(error_msg)
                    except Exception as e:
                        print(f"识别过程错误: {str(e)}")
                        
                except Exception as e:
                    print(f"监听错误: {str(e)}")
                    time.sleep(1)
                    
        except Exception as e:
            print(f"语音识别线程错误: {str(e)}")
            self.subtitle_queue.put("语音识别初始化失败")

    def radar_scan_thread(self):
        """模拟雷达扫描的线程"""
        while True:
            self.radar_angle = (self.radar_angle + 2) % 360
            # 模拟检测到的物体
            if len(self.radar_points) < 5:
                if random.random() < 0.1:  # 10%的概率生成新的点
                    angle = random.uniform(0, 360)
                    distance = random.uniform(20, 100)
                    self.radar_points.append((angle, distance))
            time.sleep(0.5)

    def gps_update_thread(self):
        """GPS数据更新线程"""
        while True:
            if self.gps_serial:
                try:
                    line = self.gps_serial.readline().decode('utf-8')
                    if line.startswith('$GPRMC'):
                        data = self.parse_gps(line)
                        if data:
                            self.current_position = data
                except:
                    pass
            time.sleep(1)

    def parse_gps(self, nmea_sentence):
        """解析GPS NMEA数据"""
        parts = nmea_sentence.split(',')
        if len(parts) >= 7 and parts[2] == 'A':  # 数据有效
            try:
                lat = float(parts[3][:2]) + float(parts[3][2:]) / 60
                lon = float(parts[5][:3]) + float(parts[5][3:]) / 60
                if parts[4] == 'S': lat = -lat
                if parts[6] == 'W': lon = -lon
                return (lat, lon)
            except:
                return None
        return None

    def detect_motion(self, frame):
        """检测画面中的运动物体"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return []
        
        # 计算当前帧与上一帧的差异
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]  # 增加阈值
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # 查找运动物体的轮廓
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_time = time.time()
        motion_points = []
        
        # 清理过期的运动点
        self.motion_points = [p for p in self.motion_points 
                            if current_time - self.last_motion_time.get(str(p), 0) < self.motion_timeout]
        
        # 处理新检测到的运动
        for contour in contours:
            if cv2.contourArea(contour) > self.min_motion_area:
                (x, y, w, h) = cv2.boundingRect(contour)
                center = (x + w//2, y + h//2)
                
                # 转换为极坐标（用于雷达显示）
                distance = math.sqrt((center[0] - frame.shape[1]/2)**2 + 
                                  (center[1] - frame.shape[0]/2)**2)
                angle = math.degrees(math.atan2(center[1] - frame.shape[0]/2,
                                              center[0] - frame.shape[1]/2))
                
                # 更新运动点的时间戳
                motion_point = (angle, distance)
                self.last_motion_time[str(motion_point)] = current_time
                motion_points.append(motion_point)
        
        # 更新运动点列表
        if motion_points:
            self.motion_points = motion_points
        
        self.prev_frame = gray
        return self.motion_points

    def detect_markers(self, frame):
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用新的API检测ArUco标记
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        return corners, ids

    def draw_cube(self, frame, corners):
        # 如果检测到标记，绘制3D立方体
        if len(corners) > 0:
            for corner in corners:
                # 使用新的API估计姿态
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corner, 
                    0.05,  # 标记的实际大小（米）
                    self.camera_matrix, 
                    self.dist_coeffs
                )
                
                # 绘制坐标轴
                cv2.drawFrameAxes(
                    frame, 
                    self.camera_matrix, 
                    self.dist_coeffs, 
                    rvec, 
                    tvec, 
                    0.03
                )

    def draw_subtitle(self, frame):
        """在左镜片上半部分绘制字幕"""
        try:
            # 如果字幕被禁用，直接返回
            if not self.subtitle_enabled:
                self.subtitle_text = ""  # 清除现有字幕
                return
                
            current_time = time.time()
            
            # 检查是否有新的字幕
            if not self.subtitle_queue.empty():
                new_text = self.subtitle_queue.get()
                if new_text:  # 确保文本不为空
                    self.subtitle_text = new_text
                    self.last_subtitle_time = current_time
                    print(f"更新字幕: {self.subtitle_text}")
            
            # 检查字幕是否应该消失
            if self.subtitle_text and current_time - self.last_subtitle_time > self.subtitle_duration:
                self.subtitle_text = ""
                return
            
            # 绘制字幕
            if self.subtitle_text:
                height, width = frame.shape[:2]
                left_lens_width = width // 2  # 左镜片宽度
                
                # 使用支持中文的字体
                fontpath = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑字体路径
                font = ImageFont.truetype(fontpath, 32)
                
                # 转换为PIL图像
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                
                # 获取文本边界框
                bbox = draw.textbbox((0, 0), self.subtitle_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # 计算位置（左镜片上部分居中）
                text_x = (left_lens_width - text_width) // 2  # 在左镜片中居中
                text_y = height // 6  # 调整到更上方的位置
                
                # 绘制背景框
                padding = 10
                cv2.rectangle(frame,
                            (text_x - padding, text_y - padding),
                            (text_x + text_width + padding, text_y + text_height + padding),
                            (0, 0, 0), -1)
                
                # 使用PIL绘制中文文本
                draw.text((text_x, text_y), 
                         self.subtitle_text, 
                         font=font, 
                         fill=(0, 255, 0))
                
                # 转换回OpenCV格式
                frame[:] = np.array(img_pil)
                
        except Exception as e:
            print(f"字幕显示错误: {str(e)}")

    def draw_navigation(self, frame):
        """在右镜片左上部分绘制导航信息"""
        try:
            if not self.navigation_active or not self.target_position:
                return
            
            height, width = frame.shape[:2]
            right_lens_start = width // 2
            
            nav_box_x = right_lens_start + 10
            nav_box_y = 20
            
            # 计算实际距离和方向
            try:
                # 确保坐标是浮点数
                current = Point(float(self.current_position[0]), float(self.current_position[1]))
                target = Point(float(self.target_position[0]), float(self.target_position[1]))
                
                # 计算距离（公里）
                dist = geodesic(current, target).kilometers
                # 计算方位角
                bearing = self.calculate_bearing(current, target)
                
                # 导航信息（使用绿色）
                nav_info = [
                    f"目的地: {self.target_position}",
                    f"距离: {dist:.1f} km",
                    f"方向: {self.get_direction(bearing)}",
                    f"角度: {bearing:.1f}°"
                ]
                
                # 绘制背景框
                max_width = 0
                for text in nav_info:
                    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    max_width = max(max_width, w)
                
                cv2.rectangle(frame,
                            (nav_box_x, nav_box_y),
                            (nav_box_x + max_width + 20, nav_box_y + len(nav_info) * 25 + 10),
                            (0, 0, 0), -1)
                
                # 绘制文本
                for i, text in enumerate(nav_info):
                    cv2.putText(frame, text,
                              (nav_box_x + 10, nav_box_y + 25 + i*25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                              
                # 绘制指向箭头
                arrow_center_x = nav_box_x + max_width + 50
                arrow_center_y = nav_box_y + 60
                arrow_length = 30
                
                angle_rad = math.radians(bearing)
                end_x = int(arrow_center_x + arrow_length * math.sin(angle_rad))
                end_y = int(arrow_center_y - arrow_length * math.cos(angle_rad))
                
                cv2.arrowedLine(frame,
                              (arrow_center_x, arrow_center_y),
                              (end_x, end_y),
                              (0, 255, 0), 2, tipLength=0.3)
                
            except Exception as e:
                print(f"导航计算错误: {str(e)}")
                cv2.putText(frame, "导航计算错误",
                          (nav_box_x + 10, nav_box_y + 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                
        except Exception as e:
            print(f"导航显示错误: {str(e)}")

    def draw_radar(self, frame):
        """在右镜片右上部分绘制雷达界面"""
        height, width = frame.shape[:2]
        right_lens_start = width // 2
        
        # 雷达中心位置（右镜片右上部分）
        radar_center = (right_lens_start + int(width/4) + 80, 100)
        radar_radius = 80  # 稍微缩小雷达尺寸
        
        # 创建透明图层
        overlay = frame.copy()
        
        # 绘制雷达轮廓（只有线条）
        cv2.circle(frame, radar_center, radar_radius, (0, 255, 0), 1)
        for r in range(1, 4):  # 距离圈
            cv2.circle(frame, radar_center, int(radar_radius * r/3), (0, 255, 0), 1)
        
        # 绘制方向标记
        directions = ['N', 'E', 'S', 'W']
        angles = [0, 90, 180, 270]
        for d, a in zip(directions, angles):
            x = int(radar_center[0] + (radar_radius + 15) * math.cos(math.radians(-a)))
            y = int(radar_center[1] + (radar_radius + 15) * math.sin(math.radians(-a)))
            cv2.putText(frame, d, (x-5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # 绘制扫描线
        angle_rad = math.radians(self.radar_angle)
        end_point = (
            int(radar_center[0] + radar_radius * math.cos(-angle_rad)),
            int(radar_center[1] + radar_radius * math.sin(-angle_rad))
        )
        cv2.line(frame, radar_center, end_point, (0, 255, 0), 1)
        
        # 绘制自身位置（绿色光点）
        cv2.circle(frame, radar_center, 4, (0, 255, 0), -1)
        cv2.circle(frame, radar_center, 6, (0, 255, 0), 1)
        
        # 绘制运动物体
        current_time = time.time()
        for point in self.motion_points:
            # 检查点是否过期
            if current_time - self.last_motion_time.get(str(point), 0) >= self.motion_timeout:
                continue
                
            angle, distance = point
            speed = self.calculate_motion_speed(point)
            is_approaching = self.is_object_approaching(point)
            
            normalized_distance = (distance / (width/2)) * radar_radius
            point_x = int(radar_center[0] + normalized_distance * math.cos(math.radians(-angle)))
            point_y = int(radar_center[1] + normalized_distance * math.sin(math.radians(-angle)))
            
            # 绘制三角形标记
            triangle_size = 6
            direction_angle = math.radians(-angle)
            
            p1 = (
                int(point_x + triangle_size * math.cos(direction_angle)),
                int(point_y + triangle_size * math.sin(direction_angle))
            )
            p2 = (
                int(point_x + triangle_size * math.cos(direction_angle + 2.6)),
                int(point_y + triangle_size * math.sin(direction_angle + 2.6))
            )
            p3 = (
                int(point_x + triangle_size * math.cos(direction_angle - 2.6)),
                int(point_y + triangle_size * math.sin(direction_angle - 2.6))
            )
            
            # 根据速度和接近状态决定颜色
            if is_approaching and speed > self.danger_speed_threshold:
                if int(current_time * 2) % 2:
                    color = (0, 0, 255)  # 红色
                else:
                    color = (0, 69, 255)  # 深红色
            else:
                color = (0, 255, 0)  # 绿色
            
            # 绘制实心三角形
            triangle_pts = np.array([p1, p2, p3], np.int32)
            cv2.fillPoly(frame, [triangle_pts], color)  # 使用fillPoly绘制实心三角形
            
            # 为危险目标添加警告圈
            if is_approaching and speed > self.danger_speed_threshold:
                warning_radius = triangle_size + 4
                cv2.circle(frame, (point_x, point_y), warning_radius, (0, 0, 255), 1)

    def calculate_motion_speed(self, point):
        """计算运动物体的速度"""
        # 这里可以通过比较前后两帧中物体的位置来计算实际速度
        # 现在使用模拟数据
        angle, distance = point
        return random.uniform(0, 30)  # 返回0-30的随机速度值

    def is_object_approaching(self, point):
        """判断物体是否在接近"""
        # 通过比较前后两帧中物体的距离变化来判断
        # 现在使用模拟数据
        angle, distance = point
        return distance < 50  # 假设距离小于50就认为是在接近

    def calculate_bearing(self, point1, point2):
        """计算两点之间的方位角"""
        lat1, lon1 = math.radians(point1.latitude), math.radians(point1.longitude)
        lat2, lon2 = math.radians(point2.latitude), math.radians(point2.longitude)
        
        d_lon = lon2 - lon1
        y = math.sin(d_lon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
        bearing = math.degrees(math.atan2(y, x))
        return (bearing + 360) % 360

    def get_direction(self, bearing):
        """将方位角转换为方向描述"""
        directions = ['北', '东北', '东', '东南', '南', '西南', '西', '西北']
        index = round(bearing / 45) % 8
        return directions[index]

    def add_obstacle(self, x, y):
        """添加障碍物"""
        self.path_planner.add_obstacle((x, y))
        # 重新规划路径
        if self.current_position and self.target_position:
            self.plan_route()
            
    def plan_route(self):
        """规划从当前位置到目标位置的路径"""
        if self.current_position and self.target_position:
            self.current_path = self.path_planner.plan_path(
                self.current_position,
                self.target_position,
                self.cap.read()[1].shape
            )
            
    def draw_path(self, frame):
        """增强版路径绘制"""
        if not self.current_path:
            return
            
        # 绘制主路径
        for i in range(len(self.current_path) - 1):
            start = self.current_path[i]
            end = self.current_path[i + 1]
            cv2.line(frame, 
                    (int(start[0]), int(start[1])),
                    (int(end[0]), int(end[1])),
                    self.path_color, 2)
            
        # 绘制路径点和转弯点标记
        for i, point in enumerate(self.current_path):
            color = (0, 0, 255) if i == 0 or i == len(self.current_path)-1 else self.path_color
            cv2.circle(frame, 
                      (int(point[0]), int(point[1])), 
                      5, color, -1)
            
            # 在转弯点添加方向指示
            if 0 < i < len(self.current_path)-1:
                prev = np.array(self.current_path[i-1])
                curr = np.array(point)
                next_point = np.array(self.current_path[i+1])
                
                # 计算转弯角度
                v1 = curr - prev
                v2 = next_point - curr
                angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
                
                if abs(angle) > 0.3:  # 如果转弯角度较大
                    cv2.circle(frame, 
                             (int(point[0]), int(point[1])),
                             8, (255, 255, 0), 1)

    def draw_obstacles(self, frame):
        """绘制障碍物"""
        for obs in self.path_planner.obstacles:
            cv2.circle(frame, 
                      (int(obs[0]), int(obs[1])), 
                      10, self.obstacle_color, -1)
            
    def set_navigation_target(self, x, y):
        """设置导航目标并规划路径"""
        self.target_position = (x, y)
        self.navigation_active = True
        self.plan_route()  # 规划新路径

    def toggle_subtitle(self, enabled):
        """切换字幕显示状态"""
        self.subtitle_enabled = enabled
        if not enabled:
            self.subtitle_text = ""  # 关闭时清除现有字幕

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # 检测运动物体
                self.motion_points = self.detect_motion(frame)
                
                # 检测标记
                corners, ids = self.detect_markers(frame)
                
                # 如果检测到标记，绘制边框和3D物体
                if ids is not None:
                    aruco.drawDetectedMarkers(frame, corners, ids)
                    self.draw_cube(frame, corners)
                
                # 按新的布局绘制界面元素
                self.draw_subtitle(frame)
                self.draw_navigation(frame)
                self.draw_radar(frame)
                
                # 绘制路径和障碍物
                self.draw_obstacles(frame)
                self.draw_path(frame)

                # 显示结果
                cv2.imshow('AR View', frame)

                # 按'q'退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        finally:
            # 清理资源
            print("正在关闭程序...")
            self.cap.release()
            cv2.destroyAllWindows()
            
            # 关闭其他资源
            if self.gps_serial:
                self.gps_serial.close()
                
            # 等待所有线程结束
            if hasattr(self, 'speech_thread'):
                self.speech_thread.join(timeout=1.0)
            if hasattr(self, 'radar_thread'):
                self.radar_thread.join(timeout=1.0)
            if hasattr(self, 'gps_thread'):
                self.gps_thread.join(timeout=1.0)

# 创建全局 AR 系统实例
_ar_system = None

def get_ar_system():
    """获取全局 AR 系统实例"""
    global _ar_system
    if _ar_system is None:
        _ar_system = ARSystem()
    return _ar_system

def process_ar_data(data):
    """
    处理AR相关的数据请求
    """
    try:
        ar_system = get_ar_system()  # 使用全局实例
        
        # 处理字幕开关请求
        if 'subtitle_enabled' in data:
            return {'status': 'success', 'subtitle_enabled': data['subtitle_enabled']}
            
        # 获取字幕设置
        if 'get_subtitle_settings' in data:
            return {'subtitle_enabled': False}  # 默认关闭
            
        # 获取当前位置
        if 'get_current_location' in data:
            return {'location': ar_system.current_position}
            
        # 获取导航设置
        if 'get_navigation_settings' in data:
            return {
                'start_location': ar_system.current_position,
                'destination': ar_system.target_position
            }
            
        # 更新导航设置
        if 'update_navigation' in data:
            ar_system.current_position = data.get('start_location', ar_system.current_position)
            ar_system.target_position = data.get('destination')
            ar_system.navigation_active = True
            
            return {
                'status': 'success',
                'navigation': {
                    'start_location': ar_system.current_position,
                    'destination': ar_system.target_position
                }
            }
            
        # 更新速度阈值
        if 'update_speed_threshold' in data:
            ar_system.danger_speed_threshold = data['threshold']
            return {
                'status': 'success',
                'speed_threshold': ar_system.danger_speed_threshold
            }
            
        return {'error': '未知的请求类型'}
        
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    ar_system = get_ar_system()  # 使用全局实例
    ar_system.run()
