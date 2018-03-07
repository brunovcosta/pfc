echo "Entrando na página de questões de concursos para pegar as matérias"
curl -sL "http://rotadosconcursos.com.br/questoes-de-concursos" \
| xmllint --html --format --xpath '//div[@class="panel panel-default"]//a[@class="list-group-item"]/@href'  - 2> /dev/null\
| sed "s/href=/\n/g" | sed "s/\"//g" > tmp_link_materias.sh 2> /dev/null

echo "Gerando script de entrar em cada página de matéria para pegar os links dos assuntos"
cat tmp_link_materias.sh | xargs -I @link_materia echo curl -sL http://rotadosconcursos.com.br@link_materia \
"| xmllint --html --format --xpath \"//div[@class='panel panel-default']//a[@class='btn btn-success btn-lg' or @class='list-group-item']/@href\"  - 2> /dev/null" \
"| sed \"s/href=/\n/g\" | sed \"s/\\\"//g\" "  > tmp_get_assuntos.sh

echo "Entrando em cada página de assunto para pegar os links de ver mais questões"
sh tmp_get_assuntos.sh |tee -a tmp_assuntos

echo "Entrando em cada link de assunto e gerando script de pegar os códigos"
cat tmp_assuntos | xargs -I @link_assunto echo curl http://rotadosconcursos.com.br@link_assunto -sL '| grep -o "+[0-9][0-9]*"' > tmp_get_codigos.sh 

echo Codigos:
sh tmp_get_codigos.sh |tee -a out

echo Abra o arquivo out
rm tmp_*.sh
