class TaskParser:
    def __init__(self, environment):
        self.environment = environment

    def parse_task(self, task):
        keywords = task.split()
        parsed_task = []

        for keyword in keywords:
            if keyword.lower() == 'or':
                parsed_task.append('|')
            elif keyword.lower() == 'and':
                parsed_task.append('&')
            elif keyword.lower() in self.environment:
                parsed_task.append(keyword.lower())

        return ' '.join(parsed_task)

if __name__ == "__main__":
    environment = ['wood', 'grass', 'toolshed', 'gold', 'iron']
    parser = TaskParser(environment)

    # Example tasks
    tasks = [
        "collect wood or collect iron and go toolshed",
        "collect wood or collect grass and dont collect gold"
    ]

    for idx, task in enumerate(tasks, start=1):
        print(f"Task {idx}: {parser.parse_task(task)}")
