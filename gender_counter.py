# gender_counter.py

class GenderCounter:
    def __init__(self):
        self.male_count = 0
        self.female_count = 0

    def update(self, gender):
        if gender.lower() == 'man' or gender.lower() == 'male':
            self.male_count += 1
        elif gender.lower() == 'woman' or gender.lower() == 'female':
            self.female_count += 1

    def get_counts(self):
        return self.male_count, self.female_count

    def reset(self):
        self.male_count = 0
        self.female_count = 0
