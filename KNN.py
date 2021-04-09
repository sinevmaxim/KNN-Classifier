import random
import time
import pylab as pl
import math
from matplotlib.colors import ListedColormap
from operator import itemgetter


class Data:
    def __init__(self, x, y, classification):
        self.x = x
        self.y = y
        self.classification = classification

    def __str__(self):
        return "x: {0}\ny: {1}\nclass: {2}".format(self.x, self.y, self.classification)


class Classifier:

    def calc_intraclass_distance(self, class_number):  # расчет внутриклассового расстояния
        def dist(a, b):
            return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)

        class_number_el = []
        for data in self.data:
            if data.classification == class_number:
                class_number_el.append(data)

        intraclass_distance = 0
        class_number_el_length = len(class_number_el)
        for i in range(class_number_el_length):
            for j in range(class_number_el_length):
                if i != j:
                    intraclass_distance += dist(class_number_el[i], class_number_el[j])

        return intraclass_distance / (class_number_el_length * (class_number_el_length - 1))

    def calc_interclass_distance(self, first_class_number, second_class_number):
        def dist(a, b):
            return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)

        first_class_number_el = []
        second_class_number_el = []
        for data in self.data:
            if data.classification == first_class_number:
                first_class_number_el.append(data)
            if data.classification == second_class_number:
                second_class_number_el.append(data)
        
        interclass_distance = 0
        first_class_number_el_length = len(first_class_number_el)
        second_class_number_el_length = len(second_class_number_el)
        for i in range(first_class_number_el_length):
            for j in range(second_class_number_el_length):
                interclass_distance += dist(first_class_number_el[i], second_class_number_el[j])
        
        return interclass_distance / (first_class_number_el_length * second_class_number_el_length)

    def generate_data(self, class_numbers, class_number_el):
        self.data = []
        self.class_numbers = class_numbers
        for class_number in range(self.class_numbers):
            center_x, center_y = random.random() * 5.0, random.random() * 5.0
            for _ in range(class_number_el):
                self.data.append(
                    Data(
                        random.gauss(center_x, 0.3),
                        random.gauss(center_y, 0.3),
                        class_number + 1,
                    )
                )
        return self.data

    def show_data(self, k_neighbours=None, show_weights=False):
        fig_1 = pl.figure(1, figsize=(20, 8.5))

        class_colormap = ListedColormap(["#FF0000", "#00FF00", "#FFFF00", "#F0FF0F", "#00FFFF", "#0FFFF0", "#000000"])
        points_data = fig_1.add_subplot(121)

        points_data.scatter(
            [self.data[i].x for i in range(len(self.data))],
            [self.data[i].y for i in range(len(self.data))],
            c=[self.data[i].classification for i in range(len(self.data))],
            cmap=class_colormap,
        )
        informativness = ''
        sum_interclasses = 0
        sum_intraclasses = 0
        for i in range(1, self.class_numbers + 1):
            for j in range(1, self.class_numbers + 1):
                if i == j:
                    pass
                else:
                    sum_interclasses += self.calc_interclass_distance(i, j)
            sum_intraclasses += self.calc_intraclass_distance(i)

        sum_interclasses = sum_interclasses / self.class_numbers**2
        sum_intraclasses = sum_intraclasses / self.class_numbers
            
        informativness += "Space informativness: {0}\n".format(sum_interclasses / sum_intraclasses)
        print("Interclass: ", sum_interclasses)
        print("Intraclass: ", sum_intraclasses)
        
        additional_info = fig_1.add_subplot(122)
        additional_info.axis("off")

        info = informativness
        if k_neighbours != None:
            k = 1
            if show_weights:
                for neighbour in k_neighbours:
                    info += "\npoint: {2}\ndistance: {0}\nclassification: {1}\nweight: {3}\n".format(
                        neighbour[0], neighbour[1], k, 1 / neighbour[0] ** 2
                    )
                    k += 1
            else:
                for neighbour in k_neighbours:
                    info += "\npoint: {2}\ndistance: {0}\nclassification: {1}\n".format(
                        neighbour[0], neighbour[1], k
                    )
                    k += 1
            
        additional_info.text(0.5, 0.5, info, ha="center", va="center")
        pl.show()

        # informativeness_text = fig_1.add_subplot(122)
        # informativness = ''
        # for i in range(1, self.class_numbers + 1):
        #     sum_interclasses = 0
        #     for j in range(1, self.class_numbers + 1):
        #         if i == j:
        #             pass
        #         else: 
        #             sum_interclasses += self.calc_interclass_distance(i, j)
        #     informativness += "\n{0} Class informativness: {1}\n".format(
        #         i, 
        #         sum_interclasses/((self.class_numbers - 1)*self.calc_intraclass_distance(i))
        #         )
        # informativeness_text.axis("off")
        # informativeness_text.text(0.5, 0.5, informativness, ha="center", va="center")
        # pl.show()

    def KNN_classify(self, data_point, k):
        def dist(a, b):
            return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)

        distances = [
            [dist(data_point, self.data[i]), self.data[i].classification]
            for i in range(len(self.data))
        ]

        k_neighbours = sorted(distances, key=lambda distance: distance[0])[0:k]
        nearest_classes_ammount = [0 for i in range(self.class_numbers + 1)]
        for neighbour in k_neighbours:
            nearest_classes_ammount[neighbour[1]] += 1

        data_point.classification = nearest_classes_ammount.index(
            max(nearest_classes_ammount)
        )

        self.data.append(data_point)
        self.show_data(k_neighbours)

    def KNN_weights_classify(self, data_point, k):
        def dist(a, b):
            return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)

        distances = [
            [dist(data_point, self.data[i]), self.data[i].classification]
            for i in range(len(self.data))
        ]

        k_neighbours = sorted(distances, key=lambda distance: distance[0])[0:k]
        nearest_classes_weights = [[0, 0] for i in range(self.class_numbers + 1)]
        for neighbour in k_neighbours:
            nearest_classes_weights[neighbour[1]][0] += 1 / (neighbour[0] ** 2)
            nearest_classes_weights[neighbour[1]][1] += 1

        nwc_copy = nearest_classes_weights.copy()
        nwc_copy.sort()
        if nwc_copy[self.class_numbers][1] != nwc_copy[self.class_numbers - 1][1]:
            classification = nearest_classes_weights.index(
                max(nearest_classes_weights, key=lambda x: x[1])
            )
        else:
            classification = nearest_classes_weights.index(
                max(nearest_classes_weights, key=lambda x: x[0])
            )

        data_point.classification = classification
        self.data.append(data_point)
        self.show_data(k_neighbours, show_weights=True)


classifier = Classifier()
classifier.generate_data(3, 40)
classifier.show_data()

for i in range(5):
    classifier.KNN_classify(
        Data(x=random.random() * 5.0, y=random.random() * 5.0, classification=0),
        k=1,
    )

classifier.generate_data(3, 40)
classifier.show_data()

for i in range(5):
    classifier.KNN_classify(
        Data(x=random.random() * 5.0, y=random.random() * 5.0, classification=0),
        k=9,
    )

classifier.generate_data(3, 40)
classifier.show_data()

for i in range(10):
    classifier.KNN_weights_classify(
        Data(x=random.random() * 5.0, y=random.random() * 5.0, classification=0),
        k=8,
    )