import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps, ImageTk, ImageFilter
import torch
import numpy as np
from MNIST.MNIST_Classificator import CNN  # ваш класс модели
import torchvision.transforms as transforms
import torch.nn.functional as F

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("MNIST Digit Recognizer - Улучшенная версия")
        self.master.geometry("800x500")
        
        # Размеры canvas - увеличиваем для лучшего рисования
        self.canvas_width = 280
        self.canvas_height = 280
        
        # Основной фрейм
        main_frame = tk.Frame(master)
        main_frame.pack(padx=10, pady=10)
        
        # Левая панель - canvas для рисования
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(left_frame, text="Нарисуйте цифру (крупно и по центру):", font=("Arial", 12)).pack()
        
        self.canvas = tk.Canvas(left_frame, width=self.canvas_width, height=self.canvas_height, 
                               bg='white', bd=2, relief=tk.SUNKEN)
        self.canvas.pack(pady=5)
        
        # Настройки кисти
        brush_frame = tk.Frame(left_frame)
        brush_frame.pack(pady=5)
        
        tk.Label(brush_frame, text="Размер кисти:").pack(side=tk.LEFT)
        self.brush_size_var = tk.IntVar(value=15)
        brush_slider = tk.Scale(brush_frame, from_=8, to=30, orient=tk.HORIZONTAL,
                               variable=self.brush_size_var, length=150)
        brush_slider.pack(side=tk.LEFT, padx=5)
        
        # Кнопки
        btn_frame = tk.Frame(left_frame)
        btn_frame.pack(pady=5)
        
        self.btn_predict = tk.Button(btn_frame, text="Распознать", command=self.predict_digit,
                                   bg='lightblue', font=("Arial", 10), width=10)
        self.btn_predict.pack(side=tk.LEFT, padx=5)
        
        self.btn_clear = tk.Button(btn_frame, text="Очистить", command=self.clear_canvas,
                                 bg='lightcoral', font=("Arial", 10), width=10)
        self.btn_clear.pack(side=tk.LEFT, padx=5)
        
        # Средняя панель - показ этапов обработки
        middle_frame = tk.Frame(main_frame)
        middle_frame.pack(side=tk.LEFT, padx=20, fill=tk.Y)
        
        tk.Label(middle_frame, text="Этапы обработки:", font=("Arial", 12, "bold")).pack()
        
        # Исходное изображение
        tk.Label(middle_frame, text="1. Исходное изображение:").pack()
        self.original_label = tk.Label(middle_frame, bd=1, relief=tk.SUNKEN)
        self.original_label.pack(pady=2)
        
        # Обрезанное изображение
        tk.Label(middle_frame, text="2. Обрезанное:").pack()
        self.cropped_label = tk.Label(middle_frame, bd=1, relief=tk.SUNKEN)
        self.cropped_label.pack(pady=2)
        
        # Центрированное изображение
        tk.Label(middle_frame, text="3. Центрированное 28x28:").pack()
        self.centered_label = tk.Label(middle_frame, bd=1, relief=tk.SUNKEN)
        self.centered_label.pack(pady=2)
        
        # Финальное изображение
        tk.Label(middle_frame, text="4. Финальное (сглаженное):").pack()
        self.final_label = tk.Label(middle_frame, bd=1, relief=tk.SUNKEN)
        self.final_label.pack(pady=2)
        
        # Правая панель - результаты
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, padx=20, fill=tk.Y)
        
        tk.Label(right_frame, text="Результаты:", font=("Arial", 12, "bold")).pack()
        
        # Результат предсказания
        self.result_frame = tk.Frame(right_frame, bd=2, relief=tk.SUNKEN, bg='white')
        self.result_frame.pack(pady=10, padx=5, fill=tk.X)
        
        self.label_result = tk.Label(self.result_frame, text="Нарисуйте цифру\nи нажмите 'Распознать'", 
                                   font=("Arial", 14), bg='white', pady=10)
        self.label_result.pack()
        
        # Показать вероятности всех классов
        self.prob_label = tk.Label(right_frame, text="Вероятности:", font=("Arial", 10))
        self.prob_label.pack(pady=(10,0))
        
        self.prob_text = tk.Text(right_frame, height=12, width=25, font=("Courier", 9))
        self.prob_text.pack(pady=5)
        
        # Советы
        tips_frame = tk.Frame(right_frame)
        tips_frame.pack(pady=10, fill=tk.X)
        
        tips_text = """Советы для лучшего распознавания:
• Рисуйте цифру крупно
• Размещайте по центру canvas
• Используйте толстую кисть (15-25)
• Рисуйте как в школьных прописях
• Избегайте слишком тонких линий"""
        
        tk.Label(tips_frame, text=tips_text, justify=tk.LEFT, 
                font=("Arial", 8), bg='lightyellow').pack(fill=tk.X)
        
        # Картинка для рисования
        self.image1 = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image1)
        
        # Переменные для рисования
        self.old_x = None
        self.old_y = None
        
        # Связываем события мыши
        self.canvas.bind('<Button-1>', self.paint_start)
        self.canvas.bind('<B1-Motion>', self.paint_move)
        self.canvas.bind('<ButtonRelease-1>', self.paint_end)
        
        # Загружаем модель
        self.load_model()
        
        # Трансформ только для изменения размера
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def load_model(self):
        """Загрузка обученной модели"""
        try:
            self.model = CNN()
            self.model.load_state_dict(torch.load("mnist_model_final.pt", map_location='cpu'))
            self.model.eval()
            print("Модель успешно загружена!")
        except FileNotFoundError:
            messagebox.showerror("Ошибка", "Файл модели 'mnist_model_final.pt' не найден!")
            self.model = None
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка загрузки модели: {str(e)}")
            self.model = None
    
    def paint_start(self, event):
        """Начало рисования"""
        self.old_x = event.x
        self.old_y = event.y
        self.paint_move(event)
    
    def paint_move(self, event):
        """Рисование при движении мыши"""
        brush_size = self.brush_size_var.get()
        
        if self.old_x and self.old_y:
            # Рисуем линию на canvas
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                  width=brush_size, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
            
            # Рисуем на PIL изображении
            self.draw.line([self.old_x, self.old_y, event.x, event.y], fill=0, width=brush_size)
            
        self.old_x = event.x
        self.old_y = event.y
    
    def paint_end(self, event):
        """Конец рисования"""
        self.old_x = None
        self.old_y = None
    
    def clear_canvas(self):
        """Очистка canvas"""
        self.canvas.delete("all")
        self.image1 = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image1)
        self.label_result.config(text="Нарисуйте цифру\nи нажмите 'Распознать'")
        
        # Очищаем все промежуточные изображения
        for label in [self.original_label, self.cropped_label, self.centered_label, self.final_label]:
            label.config(image='')
            
        self.prob_text.delete(1.0, tk.END)
    
    def center_of_mass(self, img_array):
        """Вычисление центра масс изображения"""
        h, w = img_array.shape
        total_mass = np.sum(img_array)
        
        if total_mass == 0:
            return h//2, w//2
        
        # Инвертируем для правильного расчета (черные пиксели = высокая масса)
        inverted = 255 - img_array
        
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        
        center_x = np.sum(x_coords * inverted) / np.sum(inverted)
        center_y = np.sum(y_coords * inverted) / np.sum(inverted)
        
        return int(center_y), int(center_x)
    
    def preprocess_like_mnist(self, img):
        """Предобработка изображения в стиле MNIST"""
        # 1. Инвертируем цвета
        img = ImageOps.invert(img)
        
        # Показываем исходное
        self.show_step_image(img, self.original_label, "Исходное")
        
        # 2. Находим границы объекта и обрезаем с отступом
        bbox = img.getbbox()
        if bbox is None:
            return None
            
        left, top, right, bottom = bbox
        
        # Добавляем отступ (20% от размера объекта)
        width = right - left
        height = bottom - top
        padding = max(int(0.2 * max(width, height)), 10)
        
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(img.width, right + padding)
        bottom = min(img.height, bottom + padding)
        
        # Обрезаем
        cropped = img.crop((left, top, right, bottom))
        self.show_step_image(cropped, self.cropped_label, "Обрезанное")
        
        # 3. Масштабируем, сохраняя пропорции
        # Находим максимальный размер, чтобы поместить в 20x20
        width, height = cropped.size
        max_size = 20
        
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        
        # Изменяем размер с хорошим алгоритмом
        resized = cropped.resize((new_width, new_height), Image.LANCZOS)
        
        # 4. Центрируем в изображении 28x28
        final_img = Image.new('L', (28, 28), color=0)  # Черный фон
        
        # Вычисляем позицию для центрирования
        paste_x = (28 - new_width) // 2
        paste_y = (28 - new_height) // 2
        
        final_img.paste(resized, (paste_x, paste_y))
        self.show_step_image(final_img, self.centered_label, "Центрированное")
        
        # 5. Дополнительная центровка по центру масс (как в MNIST)
        img_array = np.array(final_img)
        
        # Находим центр масс
        com_y, com_x = self.center_of_mass(img_array)
        
        # Сдвигаем к центру (14, 14)
        shift_x = 14 - com_x
        shift_y = 14 - com_y
        
        # Ограничиваем сдвиг
        shift_x = max(-3, min(3, shift_x))
        shift_y = max(-3, min(3, shift_y))
        
        if abs(shift_x) > 0 or abs(shift_y) > 0:
            # Создаем новое изображение со сдвигом
            shifted_img = Image.new('L', (28, 28), color=0)
            shifted_array = np.array(shifted_img)
            
            # Применяем сдвиг
            if shift_y >= 0 and shift_x >= 0:
                shifted_array[shift_y:, shift_x:] = img_array[:-shift_y if shift_y > 0 else 28, 
                                                              :-shift_x if shift_x > 0 else 28]
            elif shift_y >= 0 and shift_x < 0:
                shifted_array[shift_y:, :shift_x] = img_array[:-shift_y if shift_y > 0 else 28, 
                                                              -shift_x:]
            elif shift_y < 0 and shift_x >= 0:
                shifted_array[:shift_y, shift_x:] = img_array[-shift_y:, 
                                                              :-shift_x if shift_x > 0 else 28]
            else:
                shifted_array[:shift_y, :shift_x] = img_array[-shift_y:, -shift_x:]
            
            final_img = Image.fromarray(shifted_array)
        
        # 6. Легкое размытие для сглаживания
        final_img = final_img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        self.show_step_image(final_img, self.final_label, "Финальное")
        
        return final_img
    
    def show_step_image(self, img, label_widget, title):
        """Показать промежуточное изображение"""
        # Увеличиваем для отображения
        display_img = img.resize((80, 80), Image.NEAREST)
        img_tk = ImageTk.PhotoImage(display_img)
        label_widget.config(image=img_tk)
        label_widget.image = img_tk
    
    def predict_digit(self):
        """Предсказание цифры"""
        if self.model is None:
            messagebox.showerror("Ошибка", "Модель не загружена!")
            return
        
        try:
            # Проверяем, есть ли что-то нарисованное
            if self.image1.getbbox() is None:
                messagebox.showwarning("Предупреждение", "Сначала нарисуйте цифру!")
                return
            
            # Предобработка изображения
            processed_img = self.preprocess_like_mnist(self.image1.copy())
            
            if processed_img is None:
                messagebox.showwarning("Предупреждение", "Не удалось обработать изображение!")
                return
            
            # Конвертируем в тензор
            img_tensor = self.transform(processed_img).unsqueeze(0)  # [1, 1, 28, 28]
            
            # Предсказание
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = F.softmax(output, dim=1)
                pred = torch.argmax(output, dim=1).item()
                confidence = probabilities[0][pred].item()
            
            # Обновляем результат
            result_text = f"Предсказанная цифра: {pred}\nУверенность: {confidence:.1%}"
            if confidence < 0.5:
                result_text += "\n(Низкая уверенность)"
            
            self.label_result.config(text=result_text)
            
            # Показываем все вероятности
            self.show_all_probabilities(probabilities[0])
            
            print(f"Предсказание: {pred}, Уверенность: {confidence:.4f}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при предсказании: {str(e)}")
            print(f"Ошибка: {e}")
    
    def show_all_probabilities(self, probabilities):
        """Показать вероятности для всех классов"""
        self.prob_text.delete(1.0, tk.END)
        
        prob_text = "Цифра | Вероятность | Шкала\n"
        prob_text += "-" * 35 + "\n"
        
        probs_with_idx = [(i, prob.item()) for i, prob in enumerate(probabilities)]
        probs_with_idx.sort(key=lambda x: x[1], reverse=True)
        
        for i, prob in probs_with_idx:
            bar_length = int(prob * 20)  # Масштаб до 20 символов
            bar = "█" * bar_length + "░" * (20 - bar_length)
            prob_text += f"  {i}   |   {prob:.3f}   | {bar}\n"
        
        self.prob_text.insert(1.0, prob_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()