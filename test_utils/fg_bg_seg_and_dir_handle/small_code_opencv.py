#----------------------------------------------------------------------------------------------------- #
#  GMM-Mog2-基于高斯混合模型
#------------------------------------------------------------------------------------------------------#

def GMM_Mog2():

    import cv2
    cam = cv2.VideoCapture(0) # 处理视频
    fgbg = cv2.createBackgroundSubtractorMOG()
    while cam.isOpened():
        ret, frame = cam.read()
        if ret:
            fgmask = fgbg.apply(frame)
            # 通过腐蚀和膨胀过滤一些噪声
            erode = cv2.erode(fgmask, (21, 21), iterations=1)
            dilate = cv2.dilate(fgmask, (21, 21), iterations=1)
            (_, cnts, _) = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                c_area = cv2.contourArea(c)
                if c_area < 1600 or c_area > 16000:  # 过滤太小或太大的运动物体，这类误检概率比较高
                    continue
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imshow("origin", frame)
            if cv2.waitKey(1) == ord('q'):
                break
    cv2.destroyAllWindows()

#----------------------------------------------------------------------------------------------------#
# GMG-基于贝叶斯模型
#----------------------------------------------------------------------------------------------------#
def GMG():
    import cv2
    cam = cv2.VideoCapture(0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    fgbg = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=10)
    while cam.isOpened():
        ret, frame = cam.read()
        if ret:
            fgmask = fgbg.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)  # 过滤噪声
            dilate = cv2.dilate(fgmask, (21, 21), iterations=1)
            (_, cnts, _) = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                c_area = cv2.contourArea(c)
                if c_area < 1600 or c_area > 16000:  # 过滤太小或太大的运动物体，这类误检概率比较高
                    continue
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imshow("origin", frame)
            if cv2.waitKey(1) == ord('q'):
                break
    cv2.destroyAllWindows()

#------------------------------------------------------------------------------------------------------------#
# KNN

def KNN_seg():
    import cv2
    frame = ""
    bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)

    fg_mask = bs.apply(frame)

def MOG_seg():
    import cv2

    frame = ""
    history = ""
    bs = cv2.bgsegm.createBackgroundSubtractorMOG(history=history)
    bs.setHistory(history)

    fg_mask = bs.apply(frame)

def MOG2_seg():
    import cv2

    frame = ""
    history = ""
    bs = cv2.createBackgroundSubtractorMOG2(history=history, detectShadows=True)
    bs.setHistory(history)

    fg_mask = bs.apply(frame)

def GMG_seg():
    import cv2

    frame = ""
    history = ""
    bs = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=history)

    fg_mask = bs.apply(frame)

#------------------------------------------------------------------------------------------------------------#
def detect_knn_demo(video):
    import cv2, os
    camera = cv2.VideoCapture(video)
    history = 20  # 训练帧数

    bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)  # 背景减除器，设置阴影检测
    bs.setHistory(history)

    frames = 0

    while True:
        res, frame = camera.read()

        if not res:
            break

        fg_mask = bs.apply(frame)  # 获取 foreground mask

        if frames < history:
            frames += 1
            continue # 等到目标id才处理这一帧

        # 对原始帧进行膨胀去噪
        th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
        # 获取所有检测框
        # image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # py3删掉了image这个返回值

        for c in contours:
            # 获取矩形框边界坐标
            x, y, w, h = cv2.boundingRect(c)
            # 计算矩形框的面积
            area = cv2.contourArea(c)
            if 500 < area < 3000:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # cv2.imshow("detection", frame)
        filename = os.path.join("avenue_training_01_result", "avenue_training_01_detection_{}.jpg".format(str(frames)))
        cv2.imwrite(filename, frame)
        # cv2.imshow("back", dilated)
        filename = os.path.join("avenue_training_01_result", "avenue_training_01_back_{}.jpg".format(str(frames)))
        cv2.imwrite(filename, dilated)

        if cv2.waitKey(110) & 0xff == 27:
            break
    camera.release()

#--------------------------------------------------------------------------------------------------------------

def detect_knn_test_1(video):
    import cv2, os
    camera = cv2.VideoCapture(video)
    history = 20  #

    bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)  # 背景减除器，设置阴影检测
    bs.setHistory(history) # Sets the number of last frames that affect the background model

    frames = 0
    cnt = 0

    while True:
        cnt += 1
        res, frame = camera.read()

        if not res:
            break

        fg_mask = bs.apply(frame)  # 获取 foreground mask


        if frames < history:
            frames += 1
            continue # 等到时间窗口满足才开始执行下面的bg seg


        # 对原始帧进行膨胀去噪
        th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
        # 获取所有检测框
        # image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # py3删掉了image这个返回值

        for c in contours:
            # 获取矩形框边界坐标
            x, y, w, h = cv2.boundingRect(c)
            # 计算矩形框的面积
            area = cv2.contourArea(c)
            if 500 < area < 3000:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # cv2.imshow("detection", frame)
        filename = os.path.join("avenue_training_01_result", "knn_avenue_training_01_detection_{}.jpg".format(str(cnt)))
        cv2.imwrite(filename, frame)
        # cv2.imshow("back", dilated)
        filename = os.path.join("avenue_training_01_result", "knn_avenue_training_01_back_{}.jpg".format(str(cnt)))
        cv2.imwrite(filename, dilated)

        if cv2.waitKey(110) & 0xff == 27:
            break
    camera.release()

def detect_video(video, history, detector):
    import cv2, os

    camera = cv2.VideoCapture(video)
    bs = get_bs(detector, history)  # 背景减除器，设置阴影检测
    bs.setHistory(history) # Sets the number of last frames that affect the background model

    frames = 0
    cnt = 0

    while True:
        cnt += 1
        res, frame = camera.read()

        if not res:
            break

        fg_mask = bs.apply(frame)  # 获取 foreground mask


        if frames < history:
            frames += 1
            continue # 等到时间窗口满足才开始执行下面的bg seg


        # 对原始帧进行膨胀去噪
        th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
        # 获取所有检测框
        # image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # py3删掉了image这个返回值

        for c in contours:
            # 获取矩形框边界坐标
            x, y, w, h = cv2.boundingRect(c)
            # 计算矩形框的面积
            area = cv2.contourArea(c)
            if 500 < area < 3000:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        dir_name = "{}_avenue_training_01_result".format(detector)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        # cv2.imshow("detection", frame)
        filename = os.path.join(dir_name,
                                "{}_avenue_training_01_detection_{}.jpg".format(detector,str(cnt)))
        cv2.imwrite(filename, frame)
        # cv2.imshow("back", dilated)
        filename = os.path.join(dir_name,
                                "{}_avenue_training_01_back_{}.jpg".format(detector, str(cnt)))
        cv2.imwrite(filename, dilated)

        if cv2.waitKey(110) & 0xff == 27:
            break
    camera.release()

def detect_video_frames(frames_path, history, detector, dataset_name="ped2"):
    import cv2, os, glob

    out_root_path = "out_{}".format(dataset_name)
    if not os.path.exists(out_root_path):
        os.mkdir(out_root_path)

    bs = get_bs(detector, history)  # 背景减除器，设置阴影检测
    bs.setHistory(history) # Sets the number of last frames that affect the background model

    frames_dir_list = sorted(glob.glob(os.path.join(frames_path, '*')))  # 扫描所有非隐藏子目录并返回构成的list
    len_dirs = len(frames_dir_list)
    for idx in range(len_dirs):
        cur_dir = frames_dir_list[idx] # 当前子目录的path,比如 01的 path
        out_dir_path = os.path.join(out_root_path,
            "{}_{}_training_{}_result".format(detector, dataset_name,str(idx).zfill(2))
            ) # 01 对应的输出子目录
        if not os.path.exists(out_dir_path):
            os.mkdir(out_dir_path)
        cur_dir_frames_list = sorted(glob.glob(os.path.join(cur_dir, '*')))
        len_cur_dir_frames = len(cur_dir_frames_list)

        frames = 0
        cnt = 0

        for idx_frame in range(len_cur_dir_frames): # 扫描当前子目录并返回所有frame_path构成的list
            cnt += 1
            cur_frame_path = cur_dir_frames_list[idx_frame]
            frame = cv2.imread(cur_frame_path)

            fg_mask = bs.apply(frame)  # 获取 foreground mask

            if frames < history:
                frames += 1
                continue # 等到时间窗口满足才开始执行下面的bg seg

            # 对原始帧进行膨胀去噪
            th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
            th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
            dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
            # 获取所有检测框
            # image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # py3删掉了image这个返回值

            for c in contours:
                # 获取矩形框边界坐标
                x, y, w, h = cv2.boundingRect(c)
                # 计算矩形框的面积
                area = cv2.contourArea(c)
                if 500 < area < 3000:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # cv2.imshow("detection", frame)
            filename = os.path.join(out_dir_path,
                                    "{}_{}_training_detection_{}.jpg".format(dataset_name, detector,str(cnt)))
            cv2.imwrite(filename, frame)
            # cv2.imshow("back", dilated)
            filename = os.path.join(out_dir_path,
                                    "{}_{}_training_back_{}.jpg".format(dataset_name, detector, str(cnt)))
            cv2.imwrite(filename, dilated)

def get_bs(detector, history):
    import cv2

    bs = None
    if detector == "KNN":
        return cv2.createBackgroundSubtractorKNN(detectShadows=True)
    if detector == "MOG":
        return cv2.bgsegm.createBackgroundSubtractorMOG(history=history)
    if detector == "MOG2":
        return cv2.createBackgroundSubtractorMOG2(history=history, detectShadows=True)
    if detector == "GMG":
        return cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=history)


if __name__ == '__main__':
    video = '/p300/videos/avanue_training_01.avi'
    history = 20  #
    # detect_knn_demo(video)
    #
    # detect_knn_test_1(video)
    #
    detector = "MOG2"
    # detect_video(video, history, detector)
    #
    # video_frames = '/p300/ped2/training/frames'
    # dataset_name = "ped2"
    # detect_video_frames(video_frames, history, detector, dataset_name)
    #
    video_frames = '/p300/avenue/training/frames'
    dataset_name = "avenue"
    detect_video_frames(video_frames, history, detector, dataset_name)

    pass



