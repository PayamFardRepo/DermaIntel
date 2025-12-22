const en = require('./locales/en.json');
const es = require('./locales/es.json');
const fr = require('./locales/fr.json');

console.log('=== ENGLISH ===');
console.log('Title:', en.help.familyHistory.title);
console.log('Intro:', en.help.familyHistory.intro);
console.log('Step1:', en.help.familyHistory.step1);

console.log('\n=== SPANISH ===');
console.log('Title:', es.help.familyHistory.title);
console.log('Intro:', es.help.familyHistory.intro);
console.log('Step1:', es.help.familyHistory.step1);

console.log('\n=== FRENCH ===');
console.log('Title:', fr.help.familyHistory.title);
console.log('Intro:', fr.help.familyHistory.intro);
console.log('Step1:', fr.help.familyHistory.step1);

console.log('\n✅ All translations loaded successfully!');
