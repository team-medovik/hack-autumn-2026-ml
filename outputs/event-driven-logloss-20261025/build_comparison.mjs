import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const dir = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(dir, '../..');
const run = path.join(repo, 'ml/runs/event-driven-v3-logloss');
const report = JSON.parse(await fs.readFile(path.join(run, 'report.json'), 'utf8'));
const predictions = (await fs.readFile(path.join(run, 'all-predictions.jsonl'), 'utf8')).trim().split('\n').map(JSON.parse);
assert.equal(predictions.length, report.allPredictionRows);
assert.equal(new Set(predictions.map(r => r.sampleId)).size, predictions.length);
const labels = {
  newEnhanced36: 'Новая: 36 признаков', frozenLegacy24: 'Старая: 24 признака',
  refitLegacy24: '24 признака, переобучена', constantNewTrain: 'Константа нового train',
  constantOldTrain: 'Константа старого train'
};
const splits = {train:'Обучение', validation:'Выбор числа деревьев', test:'Проверка',
  purged:'Граница периодов', pending:'Исход ещё неизвестен', excluded:'Нефинансовое ограничение'};
const keys = Object.keys(labels);
const wb = Workbook.create();
const summary = wb.worksheets.add('Сводка');
const detail = wb.worksheets.add('Все прогнозы');
summary.tabColor = '#243B53';
for (const sheet of [summary, detail]) sheet.showGridLines = false;
const dark = '#243B53';
function header(sheet, range) {
  sheet.getRange(range).format = {fill:dark, font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},
    horizontalAlignment:'center', verticalAlignment:'center',wrapText:true,rowHeight:34};
}
summary.getRange('A1:F43').format = {font:{name:'Arial',size:10,color:'#172B4D'},rowHeight:22,verticalAlignment:'center'};
summary.getRange('A1:A43').format.columnWidth = 37;
summary.getRange('B1:D43').format.columnWidth = 13;
summary.getRange('E1:F43').format.columnWidth = 18;
summary.getRange('A2').values = [['Сравнение прогнозов CatBoost']];
summary.getRange('A2').format.font = {name:'Arial',size:15,bold:true};
summary.getRange('A3:F3').format.borders = {bottom:{style:'thin',color:'#CBD5E1'}};
summary.getRange('A4').values = [['Месячные VPS. Потеря: нет продления или подтверждённой замены к сроку + 5 суток.']];
summary.getRange('A5').values = [['Test: 561 снимок, 97 циклов, 7 потерь. Ранее просмотренный исследовательский набор.']];
const columns = ['Вариант','Logloss ↓','ROC-AUC ↑','Brier ↓','Средний риск','Доля потерь'];
for (const [split, start] of [['test',6],['validation',17]]) {
  summary.getRange(`A${start}:F${start}`).values = [columns];
  header(summary,`A${start}:F${start}`);
  summary.getRange(`A${start+1}:F${start+5}`).values = keys.map(key => {
    const m=report.splits[split][key]; return [labels[key],m.logloss,m.rocAuc,m.brier,m.meanProbability,m.lossRate];
  });
  summary.getRange(`B${start+1}:D${start+5}`).setNumberFormat('0.0000');
  summary.getRange(`E${start+1}:F${start+5}`).setNumberFormat('0.00%');
  summary.getRange(`A${start+1}:F${start+1}`).format.fill = '#EAF2F8';
}
summary.getRange('A13').values = [['Logloss и Brier: меньше лучше. AUC: 0,5 — случайное ранжирование.']];
summary.getRange('A14').values = [['Вес снимков одного цикла суммарно равен 1. Это не процент правильных ответов.']];
summary.getRange('A16').values = [['Validation: 805 снимков, 131 цикл, 10 потерь. Использован для выбора числа деревьев.']];
summary.getRange('A24').values = [['Logloss; depth 4; learning rate 0,05; seed 20261024. Новая: 71 дерево; контроль: 182.']];
summary.getRange('A25').values = [['Константы рассчитаны по train: новая 15,63%, прежняя 6,46%; не по test.']];
summary.getRange('A26').values = [['Генератор и старые модели не менялись. Качество на реальных клиентах не проверено.']];
summary.getRange('A28:D28').values = [['Часть','Снимки','Вес циклов','Использование']];
header(summary,'A28:D28');
const splitOrder=['train','validation','test','purged','pending','excluded'];
const last=predictions.length+1;
for (const [i,split] of splitOrder.entries()) {
  const row=29+i;
  summary.getRange(`A${row}`).values=[[split]];
  summary.getRange(`B${row}`).formulas=[[`=COUNTIFS('Все прогнозы'!$A$2:$A$${last},A${row})`]];
  summary.getRange(`C${row}`).formulas=[[`=SUMIFS('Все прогнозы'!$L$2:$L$${last},'Все прогнозы'!$A$2:$A$${last},A${row})`]];
  summary.getRange(`D${row}`).values=[[splits[split]]];
}
summary.getRange('B29:C34').setNumberFormat('#,##0');
summary.getRange('C29:C34').format.horizontalAlignment='center';
summary.getRange('A36').values = [['Train — ответы на обучающих строках, не независимая проверка качества.']];
summary.getRange('A37').values = [['Pending / excluded: факт пустой, не ноль. Прогноз сохранён только для полноты.']];
summary.getRange('A38').values = [['Циклы между частями не суммировать: purged может содержать другие снимки того же цикла.']];
summary.getRange('A40').values = [['Источник: синтетический lifecycle-v3; эксперимент event-driven-logloss-v1; 25.10.2026.']];
summary.getRange('A41').values = [['Во вкладке «Все прогнозы» фильтры по части, VPS, клиенту и фактическому исходу.']];

const headers=['Часть','VPS ID','Клиент ID','Прогноз (UTC)','Срок оплаты (UTC)','Факт: 1 = потеря',
 'Новая 36','Старая 24','Переобучена 24','Константа новая','Константа старая','Вес цикла',
 'Была в обучении','Выбор числа деревьев','Причина метки','Цена периода, EUR','Баланс, EUR',
 'Автопродление 0/1','Пополнений за 30д','Пополнено за 30д, EUR','Дней после пополнения',
 'ID снимка','ID цикла','Новая − старая, п.п.'];
detail.getRange(`A1:X${last}`).format={font:{name:'Arial',size:10,color:'#172B4D'},rowHeight:21,verticalAlignment:'center'};
detail.getRange('A1:X1').values=[headers];
detail.getRange(`A2:W${last}`).values=predictions.map(r=>{
 const f=r.observedContext;
 return [r.split,String(r.serviceId),String(r.accountId),Date.parse(r.predictionAt)/86400000+25569,Date.parse(r.dueAt)/86400000+25569,r.actualLabel,
 r.newEnhanced36,r.frozenLegacy24,r.refitLegacy24,r.constantNewTrain,r.constantOldTrain,r.cycleWeight,
 r.usedForTraining?'Да':'Нет',r.usedForModelSelection?'Да':'Нет',r.labelReason,f.price_minor/100,
 f.wallet_balance_minor===null?null:f.wallet_balance_minor/100,f.auto_prolong,f.topup_count_30d,
 f.topup_sum_30d_minor===null?null:f.topup_sum_30d_minor/100,f.days_since_topup,r.sampleId,r.cycleId];
});
detail.getRange('X2').formulas=[['=(G2-H2)*100']];
detail.getRange(`X2:X${last}`).fillDown();
detail.getRange(`A1:X${last}`).format.columnWidth=16;
detail.getRange(`D1:E${last}`).format.columnWidth=23;
detail.getRange(`O1:O${last}`).format.columnWidth=34;
detail.getRange(`V1:W${last}`).format.columnWidth=52;
detail.getRange(`D2:E${last}`).setNumberFormat('yyyy-mm-dd hh:mm:ss');
detail.getRange(`G2:K${last}`).setNumberFormat('0.00%');
detail.getRange(`L2:L${last}`).setNumberFormat('0.000000');
detail.getRange(`P2:Q${last}`).setNumberFormat('#,##0.00');
detail.getRange(`T2:U${last}`).setNumberFormat('#,##0.00');
detail.getRange(`X2:X${last}`).setNumberFormat('0.00');
const table=detail.tables.add(`A1:X${last}`,true,'ModelPredictions');
table.showFilterButton=true;
header(detail,'A1:X1');
detail.getRange('A1:X1').format.rowHeight=46;
detail.freezePanes.freezeRows(1);
detail.freezePanes.freezeColumns(3);
detail.getRange(`F2:F${last}`).conditionalFormats.add('cellIs', {operator:'equal',formula:1,
 format:{fill:'#FCE4D6',font:{color:'#9C320C',bold:true}}});

// Exercise the simple derived column, then restore the exact supplied prediction.
const first=predictions[0];
detail.getRange('G2').values=[[first.newEnhanced36+0.01]];
wb.recalculate();
assert.ok(Math.abs(detail.getRange('X2').values[0][0]-(first.newEnhanced36+0.01-first.frozenLegacy24)*100)<1e-9);
detail.getRange('G2').values=[[first.newEnhanced36]];
wb.recalculate();
for(const [i,split] of splitOrder.entries()) {
 assert.equal(summary.getRange(`B${29+i}`).values[0][0],predictions.filter(r=>r.split===split).length);
}
for(const i of [0,Math.floor(predictions.length/2),predictions.length-1]) {
 const row=i+2;
 assert.ok(Math.abs(detail.getRange(`X${row}`).values[0][0]-(predictions[i].newEnhanced36-predictions[i].frozenLegacy24)*100)<1e-9);
 assert.equal(detail.getRange(`G${row}`).values[0][0],predictions[i].newEnhanced36);
}
console.log((await wb.inspect({kind:'table',range:'Сводка!A6:F11',include:'values,formulas',tableMaxRows:6,tableMaxCols:6,maxChars:2200})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!',options:{useRegex:true,maxResults:20},summary:'formula error scan'})).ndjson);
for(const [sheetName,range,name] of [['Сводка','A1:F42','summary'],['Все прогнозы','A1:K12','predictions'],['Все прогнозы','O1:X10','context']]) {
 const preview=await wb.render({sheetName,range,scale:1.5,format:'png'});
 await fs.writeFile(path.join(dir,`${name}.png`),new Uint8Array(await preview.arrayBuffer()));
}
const xlsx=await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(path.join(dir,'all-model-predictions.xlsx'));
console.log(JSON.stringify({rows:predictions.length,file:path.join(dir,'all-model-predictions.xlsx')}));
