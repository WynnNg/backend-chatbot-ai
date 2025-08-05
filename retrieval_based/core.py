from collections import Counter
from responses import responses, blank_spot
from user_functions import preprocess, compare_overlap, pos_tag, extract_nouns, compute_similarity
import spacy
word2vec = spacy.load('en')

exit_commands = ("quit", "goodbye", "exit", "no")

class RBChatBot:
  #define .make_exit() below:
  def make_exit(self, user_message):
    for command in exit_commands:
      if command in user_message:
        print("Good Bye!")
        return True

  #define .chat() below:
  def chat(self):
    user_message = input("Hey! What would you like to know about our menu today?")
    if not user_message.strip():
      print("Please enter a question about our menu.")
      return
    while not self.make_exit(user_message):
      user_message = self.respond(user_message)

  #define .find_intent_match() below:
  def find_intent_match(self, responses, user_message):
    bow_user_message = Counter(preprocess(user_message))
    processed_responses = [Counter(preprocess(res)) for res in responses]
    similarity_list = [compare_overlap(res, bow_user_message) for res in processed_responses]
    if max(similarity_list) == 0:
      return "I'm not sure about that. Could you ask something else?"
    response_index = similarity_list.index(max(similarity_list))
    return responses[response_index]

  #define .find_entities() below:
  def find_entities(self, user_message):
    tagged_user_message = pos_tag(preprocess(user_message))
    message_nouns = extract_nouns(tagged_user_message)
    tokens = word2vec(" ".join(message_nouns))
    category = word2vec(blank_spot)
    word2vec_result = compute_similarity(tokens, category)
    word2vec_result.sort(key=lambda x: x[2])

    if len(word2vec_result) >= 1:
      return  word2vec_result[-1][0]
    return blank_spot

  #define .respond() below:
  def respond(self, user_message):
    best_response = self.find_intent_match(responses, user_message)
    entity = self.find_entities(user_message)
    print(best_response.format(entity))
    input_message = input("Do you have any other questions?")
    return input_message

#initialize ChatBot instance below:

#call .chat() method below:

RBChatBot = RBChatBot()
RBChatBot.chat()

