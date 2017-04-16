const natural = require('natural')

tokenizer = new natural.WordTokenizer()
console.log(tokenizer.tokenize("your dog has fleas.")) // [ 'your', 'dog', 'has', 'fleas' ]

puncttokenizer = new natural.TreebankWordTokenizer()
console.log(tokenizer.tokenize("my dog hasn't any fleas.")) // [ 'my', 'dog', 'hasn', 't', 'any', 'fleas' ]
console.log(puncttokenizer.tokenize("my dog hasn't any fleas.")) // [ 'my', 'dog', 'has', 'n\'t', 'any', 'fleas', '.' ]

regtokenizer = new natural.RegexpTokenizer({pattern: /\-/})
console.log(regtokenizer.tokenize("flea-dog")) // [ 'flea', 'dog' ]

wptokenizer = new natural.WordPunctTokenizer()
console.log(wptokenizer.tokenize("my dog hasn't any fleas.")) // [ 'my', 'dog', 'hasn', '\'', 't', 'any', 'fleas', '.' ]

console.log(natural.JaroWinklerDistance("dixon","dicksonx")) // 0.8133333333333332
console.log(natural.JaroWinklerDistance('not', 'same')) // 0

console.log(natural.LevenshteinDistance("ones","onez")) // 1
console.log(natural.LevenshteinDistance('one', 'one')) // 0

console.log(natural.LevenshteinDistance("ones","onez", {
    insertion_cost: 1,
    deletion_cost: 1,
    substitution_cost: 1
})) // 1

console.log(natural.DiceCoefficient('thing', 'thing')) // 1
console.log(natural.DiceCoefficient('not', 'same')) // 0

console.log(natural.PorterStemmer.stem("words")) // word
console.log(natural.PorterStemmerRu.stem("падший")) // russian
console.log(natural.PorterStemmerEs.stem("jugaría")) // spanish

natural.PorterStemmer.attach()
console.log("i am waking up to the sounds of chainsaws".tokenizeAndStem())
console.log("chainsaws".stem())

natural.LancasterStemmer.attach()
console.log("i am waking up to the sounds of chainsaws".tokenizeAndStem())
console.log("chainsaws".stem())

classifier = new natural.BayesClassifier()

classifier.addDocument('i am long qqqq', 'buy')
classifier.addDocument('buy the q\'s', 'buy')
classifier.addDocument('short gold', 'sell')
classifier.addDocument('sell gold', 'sell')
 
classifier.train()

console.log(classifier.classify('i am short silver'))
console.log(classifier.classify('i am long copper'))

console.log(classifier.getClassifications('i am long copper'))

classifier.addDocument(['sell', 'gold'], 'sell')

// POS tagger
const path = require("path");
 
const base_folder = path.join(path.dirname(require.resolve("natural")), "brill_pos_tagger");
const rulesFilename = base_folder + "/data/English/tr_from_posjs.txt";
const lexiconFilename = base_folder + "/data/English/lexicon_from_posjs.json";
const defaultCategory = 'N';
 
const lexicon = new natural.Lexicon(lexiconFilename, defaultCategory);
const rules = new natural.RuleSet(rulesFilename);
const tagger = new natural.BrillPOSTagger(lexicon, rules);
 
let sentence = ["I", "see", "the", "man", "with", "the", "telescope"];
console.log(JSON.stringify(tagger.tag(sentence)));
// [["I","NN"],["see","VB"],["the","DT"],["man","NN"],["with","IN"],["the","DT"],["telescope","NN"]] 
