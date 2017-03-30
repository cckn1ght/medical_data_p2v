
class Annotation:
    def __init__(self, tweet_id, tweet_text, tag):
        self.id = tweet_id
        self.text = tweet_text
        self.tag = tag

    def is_yes(self):
        tag = str(self.tag)
        return tag == '1' or tag == 'yes' or tag == 'y'

    def __repr__(self):
        if type(self.text) is list:
            text = " ".join(self.text)
        else:
            text = self.text
        return str(self.id) + '\t' + text + '\t' + str(self.tag)

    def get_row(self):
        return [self.id, self.text, self.tag]
        # class DataSet:
        #     data = list()

        #     def __init__(self):
        #         # self.data = data
        #         return

        #     def append(annotation):
        #         data.append(annoatation)

        #     def get_data():
        #         return data

        #     def remove_urls():
