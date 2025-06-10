# sort.py

class SimpleTracker:
    def __init__(self, max_age=5):
        self.objects = {}
        self.next_id = 1
        self.max_age = max_age

    def update(self, detections):
        updated_objects = {}

        for det in detections:
            matched = False
            for obj_id, obj in self.objects.items():
                iou = self.compute_iou(det, obj['bbox'])
                if iou > 0.3:
                    updated_objects[obj_id] = {'bbox': det, 'age': 0}
                    matched = True
                    break

            if not matched:
                updated_objects[self.next_id] = {'bbox': det, 'age': 0}
                self.next_id += 1

        # Age the objects not matched this frame
        for obj_id, obj in self.objects.items():
            if obj_id not in updated_objects:
                obj['age'] += 1
                if obj['age'] <= self.max_age:
                    # keep object alive if not too old
                    updated_objects[obj_id] = obj

        self.objects = updated_objects
        return [(obj_id, obj['bbox']) for obj_id, obj in self.objects.items()]

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        denom = float(boxAArea + boxBArea - interArea)
        if denom == 0:
            return 0
        iou = interArea / denom
        return iou


# For testing standalone
if __name__ == "__main__":
    tracker = SimpleTracker()
    test_dets = [[100, 100, 200, 200], [
        105, 105, 210, 210], [300, 300, 400, 400]]
    tracked = tracker.update(test_dets)
    print("Tracked objects:", tracked)
