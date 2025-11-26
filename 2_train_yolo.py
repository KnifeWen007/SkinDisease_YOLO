from ultralytics import YOLO


def main():
    # 推荐使用 s 版本，精度比 n 高一点，4060跑得动
    model = YOLO('yolov8s-seg.pt')

    # 开始训练
    # 注意：workers参数在Windows上如果报错，可以设为0，但通常加了main保护就不需要设为0了
    model.train(
        data='skin_multitask.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        name='yolo_task1_2',
        workers=4  # 指定多进程加载数据的线程数
    )


if __name__ == '__main__':
    # Windows 下必须把多进程代码放在这里面
    main()