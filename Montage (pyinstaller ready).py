#!/usr/bin/env python3
import os, threading, queue, pathlib
from tkinter import *
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import customtkinter as ctk
import cv2
import numpy as np
try:
    import sys, os

    def resource_path(relative_path):
        """Get absolute path to resource, works for dev and for PyInstaller exe"""
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)

    import pyi_splash
    pyi_splash.close()
except:
    pass


# ---------- Config ----------
PAGE_SIZES = {"A4": (2480,3508), "Letter": (2550,3300)}
DPI = 300
GRID_MARGIN = 40
MARGIN_OUTER = 60
INNER_CELL_MARGIN = 20

# CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

CASCADE_PATH = resource_path(os.path.join("haarcascade_frontalface_default.xml"))
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

stop_flag = False
DEFAULT_OUTPUT = os.path.join(pathlib.Path.home(), "Documents", "Montage")
icon_path = resource_path("ICON.png")


# ---------- Image Handling ----------
def detect_faces_bbox(np_img):
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    if len(faces)==0:
        h,w=gray.shape
        return int(w*0.3), int(h*0.3), int(w*0.4), int(h*0.4)
    x1=min([x for (x,y,w,h) in faces])
    y1=min([y for (x,y,w,h) in faces])
    x2=max([x+w for (x,y,w,h) in faces])
    y2=max([y+h for (x,y,w,h) in faces])
    return x1,y1,x2-x1,y2-y1

def detect_subject_bbox(np_img):
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),0)
    edges = cv2.Canny(blur,50,150)
    edges = cv2.dilate(edges,np.ones((5,5),np.uint8),1)
    contours,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h,w=gray.shape
        return int(w*0.3), int(h*0.3), int(w*0.4), int(h*0.4)
    largest = max(contours,key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(largest)
    return x,y,w,h

def place_image_in_cell(pil_img, cell_w, cell_h):
    if pil_img.width > pil_img.height:
        pil_img = pil_img.rotate(90, expand=True)
    img_w,img_h = pil_img.size
    np_img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    fx,fy,fw,fh = detect_faces_bbox(np_img)
    sx,sy,sw,sh = detect_subject_bbox(np_img)
    x1 = min(fx,sx)
    y1 = min(fy,sy)
    x2 = max(fx+fw, sx+sw)
    y2 = max(fy+fh, sy+sh)
    bbox_cx = (x1+x2)/2
    bbox_cy = (y1+y2)/2
    scale = max(cell_w/img_w, cell_h/img_h)
    new_w = int(round(img_w*scale))
    new_h = int(round(img_h*scale))
    pil_resized = pil_img.resize((new_w,new_h),resample=Image.LANCZOS)
    bbox_cx *= scale
    bbox_cy *= scale
    max_crop_x = max(new_w - cell_w,0)
    max_crop_y = max(new_h - cell_h,0)
    crop_x = int(np.clip(bbox_cx - cell_w/2,0,max_crop_x))
    crop_y = int(np.clip(bbox_cy - cell_h/2,0,max_crop_y))
    return pil_resized.crop((crop_x,crop_y,crop_x+cell_w,crop_y+cell_h))

def make_pages(task_images, page_size, rows, cols, dest_dir, progress_callback, task_index):
    global stop_flag
    MAX_PER_PAGE = rows * cols
    page_w, page_h = page_size
    TOTAL_H_MARGIN = (cols - 1) * GRID_MARGIN + 2 * MARGIN_OUTER + 2 * INNER_CELL_MARGIN * cols
    TOTAL_V_MARGIN = (rows - 1) * GRID_MARGIN + 2 * MARGIN_OUTER + 2 * INNER_CELL_MARGIN * rows
    CELL_W = (page_w - TOTAL_H_MARGIN)//cols
    CELL_H = (page_h - TOTAL_V_MARGIN)//rows
    os.makedirs(dest_dir, exist_ok=True)

    for pidx in range(0, len(task_images), MAX_PER_PAGE):
        if stop_flag: break
        batch = task_images[pidx:pidx+MAX_PER_PAGE]
        canvas = Image.new("RGB",(page_w,page_h),(255,255,255))

        for i, img_path in enumerate(batch):
            if stop_flag: break
            try:
                pil_img = Image.open(img_path).convert("RGB")
            except:
                continue

            cell_img = place_image_in_cell(pil_img, CELL_W, CELL_H)
            row = i // cols
            col = i % cols
            x = MARGIN_OUTER + col*(CELL_W + GRID_MARGIN + 2*INNER_CELL_MARGIN) + INNER_CELL_MARGIN
            y = MARGIN_OUTER + row*(CELL_H + GRID_MARGIN + 2*INNER_CELL_MARGIN) + INNER_CELL_MARGIN
            canvas.paste(cell_img, (int(x), int(y)))

            # Update progress bar for this task
            if progress_callback:
                progress_callback(pidx + i + 1, len(task_images))

        out_name = os.path.join(dest_dir, f"task{task_index:02d}_page_{(pidx//MAX_PER_PAGE)+1:03d}.png")
        canvas.save(out_name, dpi=(DPI, DPI))


def create_task_thumbnail(image_paths, size=(60, 60)):
    """
    Creates a 2x2 grid thumbnail from up to 4 images,
    scaling/cropping each to completely fill its cell.
    """
    from PIL import Image, ImageTk

    def scale_and_crop(img, cell_w, cell_h):
        img_w, img_h = img.size
        scale = max(cell_w / img_w, cell_h / img_h)
        new_w = int(round(img_w * scale))
        new_h = int(round(img_h * scale))
        img_resized = img.resize((new_w, new_h), resample=Image.LANCZOS)

        crop_x = (new_w - cell_w) // 2
        crop_y = (new_h - cell_h) // 2
        return img_resized.crop((crop_x, crop_y, crop_x + cell_w, crop_y + cell_h))

    thumb = Image.new("RGB", size, (200, 200, 200))  # base gray background
    positions = [
        (0, 0),
        (size[0] // 2, 0),
        (0, size[1] // 2),
        (size[0] // 2, size[1] // 2)
    ]
    cell_w, cell_h = size[0] // 2, size[1] // 2

    for i, img_path in enumerate(image_paths[:4]):
        try:
            img = Image.open(img_path)
            cell_img = scale_and_crop(img, cell_w, cell_h)
            x, y = positions[i]
            thumb.paste(cell_img, (x, y))
        except:
            continue

    return ImageTk.PhotoImage(thumb)


def center_window(win):
    win.update_idletasks()  # Ensure size info is up to date
    width = win.winfo_width()
    height = win.winfo_height()
    screen_width = win.winfo_screenwidth()
    screen_height = win.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    win.geometry(f"+{x}+{y}")




# ---------- Task ----------
class Task:
    def __init__(self, images,page_type,rows,cols,subfolder_name=None):
        self.images = images
        self.page_type = page_type
        self.rows = rows
        self.cols = cols
        self.status = "Pending"
        self.subfolder_name = subfolder_name
        self.thumbnail = None
        if images:
            try:
                pil_img = Image.open(images[0])
                pil_img.thumbnail((60,60))
                self.thumbnail = ImageTk.PhotoImage(pil_img)
            except: self.thumbnail = None

# ---------- GUI ----------
class MontageGUI:
    def __init__(self,master):
        self.master=master
        master.title("Modern Montage")
        master.geometry("850x400")
        center_window(master)
        self.tasks=[]
        self.dest_dir=StringVar(value=DEFAULT_OUTPUT)
        self.progress_queue=queue.Queue()
        self.active_task=None
        self.task_thread=None
        self.selected_task_index=None
        self.thumbnail_cache = {}
        self.task_frames = []
        self.create_widgets()
        self.master.after(100,self.update_progress)


        icon = Image.open(icon_path)              # load image with PIL
        icon_tk = ImageTk.PhotoImage(icon)         # convert to Tkinter image
        master.iconphoto(False, icon_tk)           # set window icon
        self.icon_ref = icon_tk   # prevent garbage collection

     
        self.menu_bar = Menu(master)
        master.config(menu=self.menu_bar)
        # File menu
        file_menu = Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Add Task", command=self.add_task_with_images)
        file_menu.add_command(label="Select Destination", command=self.select_dest)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=master.quit)
        self.menu_bar.add_cascade(label="File", menu=file_menu)

        # Edit menu
        edit_menu = Menu(self.menu_bar, tearoff=0)
        edit_menu.add_command(label="Clear All Tasks", command=self.clear_all_tasks)
        self.menu_bar.add_cascade(label="Edit", menu=edit_menu)

        # Help menu
        help_menu = Menu(self.menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=lambda: self.show_about())
        self.menu_bar.add_cascade(label="Help", menu=help_menu)



    def show_about(self):
        about_win = Toplevel(self.master)
        about_win.overrideredirect(True)  # remove top bar
        about_win.resizable(False, False)

        # Center the window
        win_width = 600
        win_height = 425
        screen_width = about_win.winfo_screenwidth()
        screen_height = about_win.winfo_screenheight()
        x = (screen_width // 2) - (win_width // 2)
        y = (screen_height // 2) - (win_height // 2)
        about_win.geometry(f"{win_width}x{win_height}+{x}+{y}")

        about_win.bind("<Control-q>", lambda e: about_win.overrideredirect(False), about_win.resizable(False,True))

        main_frame = ttk.Frame(about_win, relief=RAISED, borderwidth=2, padding=10)
        main_frame.pack(fill=BOTH, expand=True)

        info = {
            "Internal name": "Photo collage grid",
            "Version": "1.1.5",
            "Developer": "Ahnaf Tauseef Khan (Jashore, Khulna, Bangladesh)",
            "Phone": "+088 01988947838",
            "Email": "tawsifa7@gmail.com",
            "GitHub": "https://github.com/AhnafTouseef",
            "Facebook": "https://www.facebook.com/tshiba.khan",
            "Developer's comment": (
                "This is a software project made on 08 September, 2025 as part of an experimental project for "
                "my friend Shakir Ahmed Bulbul (Khulna, Bangladesh). The primary goal of the software is to "
                "automate the process of arranging multiple images on an A4 paper in a defined grid without "
                "losing its main subject. It utilizes Python and its extensive libraries to perform all tasks, "
                "GUI management, import, and export. There is more room for development."
            )
        }

        def make_clickable(label, url):
            import webbrowser
            label.bind("<Button-1>", lambda e: webbrowser.open(url))
            label.config(fg="blue", cursor="hand2")

        for row, (key, value) in enumerate(info.items()):
            lbl_key = Label(main_frame, text=f"{key}:", anchor="w", justify=LEFT, font=("Arial", 10, "bold"), width=20)
            lbl_key.grid(row=row, column=0, sticky="nw", padx=(0, 10), pady=2)

            lbl_value = Label(main_frame, text=value, anchor="w", justify=LEFT, wraplength=400)
            lbl_value.grid(row=row, column=1, sticky="nw", pady=2)

            # Make links clickable
            if key in ("GitHub", "Facebook"):
                make_clickable(lbl_value, value)
            elif key == "Email":
                make_clickable(lbl_value, f"mailto:{value}")
        
        # OK button at bottom
        btn_ok = ttk.Button(main_frame, text="OK", command=about_win.destroy)
        btn_ok.grid(row=len(info), column=0, columnspan=2, pady=15)

        img = Image.open(resource_path("ICON.png"))
        img = img.resize((95, 95), Image.LANCZOS)  # width x height
        tk_img = ImageTk.PhotoImage(img)

        def config():
            img = Image.open(resource_path("Author1.png"))
            img = img.resize((100, 100), Image.LANCZOS)  # width x height
            tk_img = ImageTk.PhotoImage(img)
            btn_image.configure(image=tk_img)

        btn_image = ctk.CTkButton(main_frame, text="", image= tk_img, width=100,height=100, command=config, corner_radius=0)
        btn_image.grid(row=8, column=0, rowspan=1, sticky="nw")

        img_auth = Image.open(resource_path("Author2.png")).resize((400,400))
        IMG_AUTH = ImageTk.PhotoImage(img_auth)
        AUTH_image = ctk.CTkButton(about_win, text="", image= IMG_AUTH,  command=config, corner_radius=0)
        AUTH_image.pack()
        




    def add_task_with_images(self):
        from tkinter import filedialog
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
        )
        if file_paths:
            self.open_intermediate(list(file_paths))

    def open_task_dir(self, task, base_dir):
        if task.subfolder_name:
            folder_path = os.path.join(base_dir, task.subfolder_name)
        else:
            folder_path = base_dir
        if os.path.isdir(folder_path):
            import sys, subprocess
            if sys.platform == "win32":
                os.startfile(folder_path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder_path])
            else:
                subprocess.Popen(["xdg-open", folder_path])



    def create_widgets(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("green.Horizontal.TProgressbar", troughcolor='white', background='green')
        control_frame = ttk.Frame(self.master)
        control_frame.pack(fill=X,padx=10,pady=5)
        ttk.Button(control_frame,text="Add Task",command=self.add_task_with_images).pack(side=LEFT, padx=5)
        ttk.Button(control_frame,text="Start All",command=self.start_all_tasks).pack(side=LEFT,padx=5)
        ttk.Button(control_frame,text="Clear All Tasks",command=self.clear_all_tasks).pack(side=LEFT,padx=5)
        ttk.Entry(control_frame,textvariable=self.dest_dir,width=50).pack(side=LEFT,padx=5, fill=X, expand=True)
        ttk.Button(control_frame,text="Select Destination",command=self.select_dest).pack(side=RIGHT,padx=5)

        # Task cards canvas
        self.canvas_frame = Frame(self.master)
        self.canvas_frame.pack(fill=BOTH,expand=True,padx=10,pady=5)
        self.canvas = Canvas(self.canvas_frame)
        self.scrollbar = ttk.Scrollbar(self.canvas_frame, orient=VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=RIGHT, fill=Y)
        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.cards_frame = ttk.Frame(self.canvas)
        def on_canvas_resize(event):
            self.canvas.itemconfig(self.canvas_window, width=event.width)

        self.canvas_window = self.canvas.create_window((0,0), window=self.cards_frame, anchor="nw")
        self.canvas.bind("<Configure>", on_canvas_resize)

        self.cards_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        

    # ---------- Task List Refresh ----------
    def refresh_task_list(self):
        for w in self.cards_frame.winfo_children():
            w.destroy()
        self.task_frames.clear()

        for idx, task in enumerate(self.tasks):
            frame = ttk.Frame(self.cards_frame, relief=RIDGE, borderwidth=2, padding=5)
            frame.pack(fill=X, pady=2)
            self.task_frames.append(frame)

            # Highlight if selected
            if self.selected_task_index == idx:
                frame.config(style='Selected.TFrame')
            else:
                frame.config(style='TFrame')

            # Subframes
            subframe_image = ttk.Frame(frame)
            subframe_info = ttk.Frame(frame)
            subframe_buttons = ttk.Frame(frame)

            # Thumbnail (or placeholder)
            task.thumbnail = create_task_thumbnail(task.images)
            lbl_img = Label(subframe_image, image=task.thumbnail, width=60, height=60)

            lbl_img.pack(side=LEFT, padx=5)

            # Per-task progress bar
            task.progressbar = ttk.Progressbar(subframe_info, orient=HORIZONTAL, length=430, mode='determinate')
            task.progressbar.pack(side=TOP, anchor=W, padx=5, pady=2)

            if task.status == "Done":
                task.progressbar['value'] = task.progressbar['maximum']  # optional: show full
            else:
                task.progressbar['value'] = 0
                task.progressbar['maximum'] = len(task.images)

            style = ttk.Style()
            style.configure("green.Horizontal.TProgressbar", foreground='green', background='green')
            task.progressbar.config(style="green.Horizontal.TProgressbar")
            
            # Info
            dest_path = self.dest_dir.get()
            if task.subfolder_name:
                dest_path = os.path.join(dest_path, task.subfolder_name)

            lbl_text1 = Label(subframe_info, text=(f"{len(task.images)} images | "f"{task.page_type} | "f"{task.rows}x{task.cols} | "),width=60,anchor=W,justify=LEFT,wraplength=600)
            lbl_text1.pack(side=TOP, padx=5)

            lbl_text2 = Label(subframe_info,text=(f"{dest_path} | "f"{task.status}"),width=60,anchor=W,justify=LEFT,wraplength=600)
            lbl_text2.pack(side=TOP, padx=5)

            # Buttons
            ttk.Button(subframe_buttons, text="Edit", command=lambda i=idx: self.edit_task(i)).pack(side=LEFT, padx=5)
            ttk.Button(subframe_buttons, text="Remove", command=lambda i=idx: self.remove_task(i)).pack(side=LEFT, padx=5)
            if task.status == "Done":
                btn_text = "Open Folder"
                btn_command = lambda t=task: self.open_task_dir(t, self.dest_dir.get())
            else:
                btn_text = "Start"
                btn_command = lambda i=idx: self.run_task(i)

            start_btn = ttk.Button(subframe_buttons, text=btn_text, command=btn_command)
            start_btn.pack(side=LEFT, padx=5)
            task.start_btn = start_btn



            # pack subframes
            subframe_image.pack(side=LEFT)
            subframe_info.pack(side=LEFT)
            subframe_buttons.pack(side=RIGHT)


            def _bind_mousewheel(widget):
                if os.name == 'nt':          # Windows
                    widget.bind_all("<MouseWheel>", lambda e: widget.yview_scroll(int(-1*(e.delta/120)), "units"))
                elif os.name == 'mac':       # macOS (rarely 'mac' — safer to check platform if you need)
                    widget.bind_all("<MouseWheel>", lambda e: widget.yview_scroll(int(-1*e.delta), "units"))
                else:                        # Linux/others (X11)
                    widget.bind_all("<Button-4>", lambda e: widget.yview_scroll(-1, "units"))
                    widget.bind_all("<Button-5>", lambda e: widget.yview_scroll(1, "units"))


            # bind mouse wheel only while pointer is over the canvas (prevents global interference)
            def _on_enter(e):
                _bind_mousewheel(self.canvas)
            def _on_leave(e):
                try:
                    self.canvas.unbind_all("<MouseWheel>")
                    self.canvas.unbind_all("<Button-4>")
                    self.canvas.unbind_all("<Button-5>")
                except Exception:
                    pass

            self.canvas.bind("<Enter>", _on_enter)
            self.canvas.bind("<Leave>", _on_leave)


    def select_task(self,index):
        self.selected_task_index=index
        for i,frame in enumerate(self.task_frames):
            frame.config(style="TFrame" if i!=index else "Selected.TFrame")
        style = ttk.Style()
        style.configure("Selected.TFrame", background="#cfffcf")

    def start_selected_task(self):
        if self.selected_task_index is not None:
            self.run_task(self.selected_task_index)

    def start_all_tasks(self):
        # Disable the button while processing

        def run_all():
            for idx, task in enumerate(self.tasks):
                if task.status == "Done":
                    continue  # skip finished tasks
                task.status = "Processing"
                self.refresh_task_list()
                self.run_task(idx)

                # Wait until this task finishes
                while task.status == "Processing":
                    self.master.update()
            

        threading.Thread(target=run_all, daemon=True).start()


    def clear_all_tasks(self):
        global stop_flag
        # stop_flag=True
        self.tasks.clear()
        self.refresh_task_list()
        # self.progress['value']=0

    def select_dest(self):
        directory = filedialog.askdirectory()
        if directory:
            self.dest_dir.set(directory)

    # ---------- Run Task ----------
    def run_task(self, index):
        task = self.tasks[index]
        task.status = "Processing"
        self.refresh_task_list()

        page_size = PAGE_SIZES.get(task.page_type, PAGE_SIZES["A4"])

        # Choose destination directory, respecting per-task subfolder
        dest_dir = self.dest_dir.get()
        if task.subfolder_name:
            # sanitize/normalize just in case
            dest_dir = os.path.join(dest_dir, os.path.basename(task.subfolder_name))

        # progress callback to safely update the GUI progressbar
        def progress_callback(current, total):
            # update in main thread
            self.master.after(0, lambda: task.progressbar.config(value=current, maximum=total))

        # worker function
        def worker():
            try:
                make_pages(task.images, page_size, task.rows, task.cols, dest_dir, progress_callback, index + 1)
            finally:
                # on finish, update status and clear progress in main thread
                def finish_updates():
                    task.status = "Done"
                    try:
                        task.progressbar.config(value=0)
                    except Exception:
                        pass
                    self.refresh_task_list()

                    # Update button to "Open Folder" if task has subfolder
                    if hasattr(task, 'start_btn'):
                        task.start_btn.config(
                            text="Open Folder",
                            command=lambda t=task, d=self.dest_dir.get(): self.open_task_dir(t, d)
                        )


                self.master.after(0, finish_updates)

        threading.Thread(target=worker, daemon=True).start()


    # ---------- Intermediate Window ----------
    def open_intermediate(self, images=None):
        if images is None:
            images = []

        selected_indices=set()
        win = Toplevel(self.master)
        win.title("Setup Task")
        win.transient(self.master)
        win.grab_set()
        win.lift()
        win.geometry("550x450")

        icon = Image.open(icon_path)              # load image with PIL
        icon_tk = ImageTk.PhotoImage(icon)         # convert to Tkinter image
        win.iconphoto(False, icon_tk)           # set window icon
        self.icon_ref = icon_tk                # prevent garbage collection


        center_window(win)
        # Page type
        page_type_var = StringVar(value="A4")
        ttk.Label(win,text="Page Type:").pack(anchor=W)
        ttk.Combobox(win,textvariable=page_type_var,values=list(PAGE_SIZES.keys())).pack(anchor=W)
        # Grid
        rows_var = IntVar(value=2)
        cols_var = IntVar(value=2)
        grid_frame = ttk.Frame(win)
        grid_frame.pack(anchor=W,pady=5)
        ttk.Label(grid_frame,text="Rows:").pack(side=LEFT)
        ttk.Entry(grid_frame,textvariable=rows_var,width=5).pack(side=LEFT,padx=5)
        ttk.Label(grid_frame,text="Cols:").pack(side=LEFT)
        ttk.Entry(grid_frame,textvariable=cols_var,width=5).pack(side=LEFT,padx=5)
        # Subfolder option
        subfolder_var = StringVar()
        chk = ttk.Checkbutton(win, text="Use subfolder for output", variable=subfolder_var, onvalue="1", offvalue="", command=lambda: entry_subfolder.configure(state=NORMAL if subfolder_var.get()=="1" else DISABLED))
        chk.pack(anchor=W, pady=5)
        entry_subfolder = ttk.Entry(win)
        entry_subfolder.pack(anchor=W, padx=20)
        entry_subfolder.configure(state=DISABLED)

                # --- Thumbnails (fixed scrollbar position) ---
        # container to hold canvas + scrollbar so scrollbar stays next to canvas
        thumb_container = ttk.Frame(win)
        thumb_container.pack(fill=BOTH, expand=True, padx=5, pady=5)

        thumb_canvas = Canvas(thumb_container, height=250)
        thumb_canvas.pack(side=LEFT, fill=BOTH, expand=True)

        thumb_scrollbar = ttk.Scrollbar(thumb_container, orient=VERTICAL, command=thumb_canvas.yview)
        thumb_scrollbar.pack(side=RIGHT, fill=Y)

        thumb_canvas.configure(yscrollcommand=thumb_scrollbar.set)

        thumb_frame = ttk.Frame(thumb_canvas)
        thumb_window = thumb_canvas.create_window((0, 0), window=thumb_frame, anchor='nw')

        # keep canvas scrollregion up-to-date when inner frame changes
        def _on_frame_configure(event):
            thumb_canvas.configure(scrollregion=thumb_canvas.bbox("all"))
        thumb_frame.bind("<Configure>", _on_frame_configure)

        # make the inner frame match canvas width (so widgets wrap correctly)
        def _on_canvas_configure(event):
            thumb_canvas.itemconfig(thumb_window, width=event.width)
        thumb_canvas.bind("<Configure>", _on_canvas_configure)

        # optional: mouse wheel scrolling when pointer is over the canvas
        def _bind_mousewheel(widget):
            if os.name == 'nt':          # Windows
                widget.bind_all("<MouseWheel>", lambda e: widget.yview_scroll(int(-1*(e.delta/120)), "units"))
            elif os.name == 'mac':       # macOS (rarely 'mac' — safer to check platform if you need)
                widget.bind_all("<MouseWheel>", lambda e: widget.yview_scroll(int(-1*e.delta), "units"))
            else:                        # Linux/others (X11)
                widget.bind_all("<Button-4>", lambda e: widget.yview_scroll(-1, "units"))
                widget.bind_all("<Button-5>", lambda e: widget.yview_scroll(1, "units"))

        # bind mouse wheel only while pointer is over the canvas (prevents global interference)
        def _on_enter(e):
            _bind_mousewheel(thumb_canvas)
        def _on_leave(e):
            try:
                thumb_canvas.unbind_all("<MouseWheel>")
                thumb_canvas.unbind_all("<Button-4>")
                thumb_canvas.unbind_all("<Button-5>")
            except Exception:
                pass

        thumb_canvas.bind("<Enter>", _on_enter)
        thumb_canvas.bind("<Leave>", _on_leave)

        # finally populate thumbnails
        self.refresh_thumbnails(images, selected_indices, thumb_frame)


        from datetime import datetime

        def add_task_final():
            if subfolder_var.get() == "1":
                entered_name = entry_subfolder.get().strip()
                if entered_name:
                    subfolder_name = entered_name
                else:
                    # Generate folder name: TaskXX_YYYYMMDD_HHMMSS
                    subfolder_name = f"Task{len(self.tasks)+1:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                subfolder_name = None

            if images:
                task = Task(list(images), page_type_var.get(), rows_var.get(), cols_var.get(), subfolder_name)
                self.tasks.append(task)
                self.refresh_task_list()
                win.destroy()


        btn_frame = ttk.Frame(win)
        btn_frame.pack(side=BOTTOM,pady=5)
        ttk.Button(btn_frame,text="Add Images",command=lambda:self.add_images(images,selected_indices,thumb_frame)).pack(side=LEFT,padx=5)
        ttk.Button(btn_frame,text="Add Task",command=add_task_final).pack(side=LEFT,padx=5)

    # ---------- Thumbnail Management ----------
    def refresh_thumbnails(self, images, selected_indices, frame):
        for w in frame.winfo_children(): w.destroy()
        for idx, img_path in enumerate(images):
            try:
                if img_path not in self.thumbnail_cache:
                    pil_img = Image.open(img_path)
                    pil_img.thumbnail((80, 80))
                    self.thumbnail_cache[img_path] = ImageTk.PhotoImage(pil_img)
                tkimg = self.thumbnail_cache[img_path]
                
                # Image label
                lbl = Label(frame, image=tkimg, relief=RIDGE, borderwidth=2)
                lbl.image = tkimg
                lbl.grid(row=idx//6, column=idx%6, padx=2, pady=2)

                # Selection border
                def on_click(event, ix=idx):
                    selected_indices.clear()
                    selected_indices.add(ix)
                    self.refresh_thumbnails(images, selected_indices, frame)
                lbl.bind("<Button-1>", on_click)
                if idx in selected_indices:
                    lbl.config(relief=SUNKEN, highlightbackground="red", highlightthickness=3)

                # Cross button
                btn = Button(frame, text="✕", command=lambda ix=idx: self.remove_image(ix, images, selected_indices, frame))
                btn.place(in_=lbl, relx=1, rely=0, anchor="ne")

            except: continue

    def remove_image(self, idx, images, selected_indices, frame):
        images.pop(idx)
        selected_indices.clear()
        self.refresh_thumbnails(images, selected_indices, frame)



    def toggle_select(self,index,selected_indices,label):
        if index in selected_indices:
            selected_indices.remove(index)
            label.config(relief=SUNKEN)
        else:
            selected_indices.add(index)
            label.config(relief=RAISED)

    def add_images(self,images,selected_indices,frame):
        files = filedialog.askopenfilenames(filetypes=[("Images","*.png *.jpg *.jpeg")])
        if files:
            images.extend(files)
            self.refresh_thumbnails(images,selected_indices,frame)

    def remove_selected_image(self,images,selected_indices,frame):
        for idx in sorted(selected_indices,reverse=True):
            if idx<len(images):
                images.pop(idx)
        selected_indices.clear()
        self.refresh_thumbnails(images,selected_indices,frame)

    def edit_task(self, index):
        task = self.tasks[index]
        images = list(task.images)
        selected_indices=set()
        win = Toplevel(self.master)
        win.title("Setup Task")
        win.transient(self.master)
        win.grab_set()
        win.lift()
        win.geometry("550x450")

        icon = Image.open(icon_path)              # load image with PIL
        icon_tk = ImageTk.PhotoImage(icon)         # convert to Tkinter image
        win.iconphoto(False, icon_tk)           # set window icon
        self.icon_ref = icon_tk   # prevent garbage collection


        center_window(win)
        # Page type
        page_type_var = StringVar(value=task.page_type)

        ttk.Label(win,text="Page Type:").pack(anchor=W)
        ttk.Combobox(win,textvariable=page_type_var,values=list(PAGE_SIZES.keys())).pack(anchor=W)
        # Grid
        rows_var = IntVar(value=task.rows)
        cols_var = IntVar(value=task.cols)
        grid_frame = ttk.Frame(win)
        grid_frame.pack(anchor=W,pady=5)
        ttk.Label(grid_frame,text="Rows:").pack(side=LEFT)
        ttk.Entry(grid_frame,textvariable=rows_var,width=5).pack(side=LEFT,padx=5)
        ttk.Label(grid_frame,text="Cols:").pack(side=LEFT)
        ttk.Entry(grid_frame,textvariable=cols_var,width=5).pack(side=LEFT,padx=5)

        # Subfolder option
        subfolder_var = StringVar(value="1" if task.subfolder_name else "")
        chk = ttk.Checkbutton(
            win,
            text="Use subfolder for output",
            variable=subfolder_var,
            onvalue="1",
            offvalue="",
            command=lambda: entry_subfolder.configure(
                state=NORMAL if subfolder_var.get()=="1" else DISABLED
            )
        )
        chk.pack(anchor=W, pady=5)

        entry_subfolder = ttk.Entry(win)
        entry_subfolder.pack(anchor=W, padx=20)
        entry_subfolder.configure(state=NORMAL if task.subfolder_name else DISABLED)
        if task.subfolder_name:
            entry_subfolder.insert(0, task.subfolder_name)


        # Thumbnails
        thumb_canvas = Canvas(win,height=250)
        thumb_canvas.pack(fill=BOTH,expand=True,padx=5,pady=5)
        scrollbar = ttk.Scrollbar(win, orient=VERTICAL, command=thumb_canvas.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        thumb_canvas.configure(yscrollcommand=scrollbar.set)
        thumb_frame = ttk.Frame(thumb_canvas)
        thumb_canvas.create_window((0,0),window=thumb_frame,anchor='nw')
        thumb_frame.bind("<Configure>", lambda e: thumb_canvas.configure(scrollregion=thumb_canvas.bbox("all")))
        self.refresh_thumbnails(images,selected_indices,thumb_frame)


        def save_changes():
            task.images = list(images)
            task.page_type = page_type_var.get()
            task.rows = rows_var.get()
            task.cols = cols_var.get()

            # Reset status and progress
            task.status = "Pending"
            if hasattr(task, "progressbar"):
                task.progressbar['value'] = 0
                task.progressbar['maximum'] = len(task.images)

            # Handle thumbnail
            if task.images:
                try:
                    pil_img = Image.open(task.images[0])
                    pil_img.thumbnail((60,60))
                    task.thumbnail = ImageTk.PhotoImage(pil_img)
                except:
                    task.thumbnail = None

            # Handle subfolder logic
            if subfolder_var.get() == "1":
                entered_name = entry_subfolder.get().strip()
                if entered_name:
                    task.subfolder_name = entered_name
                else:
                    from datetime import datetime
                    task.subfolder_name = f"Task{index+1:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                task.subfolder_name = None

            # Refresh task list (this will rebuild buttons and progress bar)
            self.refresh_task_list()
            win.destroy()






        btn_frame = ttk.Frame(win)
        btn_frame.pack(side=BOTTOM,pady=5)
        ttk.Button(btn_frame,text="Add Images",command=lambda:self.add_images(images,selected_indices,thumb_frame)).pack(side=LEFT,padx=5)
        ttk.Button(btn_frame,text="Save Task",command=save_changes).pack(side=LEFT,padx=5)

    def remove_task(self, index):
        self.tasks.pop(index)
        self.refresh_task_list()


    # ---------- Progress ----------
    def update_progress(self):
        try:
            while True:
                msg,val = self.progress_queue.get_nowait()
                if msg=="Processing":
                    cur,total = val
                    self.progress['maximum']=total
                    self.progress['value']=cur
                elif msg=="PageSaved":
                    pass
                elif msg=="Done":
                    self.progress['value']=0
        except queue.Empty:
            pass
        self.master.after(100,self.update_progress)

if __name__=="__main__":
    root=Tk()
    MontageGUI(root)
    center_window(root)
    root.mainloop()
