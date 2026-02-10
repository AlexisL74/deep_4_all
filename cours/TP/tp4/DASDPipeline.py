from student_model import StudentModel
from Promptotron import Promptotron

class DASPipeline:
    def __init__(self, teacher_model: Promptotron, student_model: StudentModel) -> None:
        self.teacher_model = teacher_model
        self.student_model = student_model