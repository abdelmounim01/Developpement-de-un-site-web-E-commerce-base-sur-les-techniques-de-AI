import { Component, OnInit } from '@angular/core';
import Chart from 'node_modules/chart.js/dist/Chart.js'
import { ProjetService } from '../../service/ImplementationService/projet.service'
import { HttpClient } from '@angular/common/http'
import { Info } from '../../service/Interfaces/info'
import { User } from 'src/app/service/Interfaces/user';
import { Task } from 'src/app/service/Interfaces/task';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css']
})
export class DashboardComponent implements OnInit {
  public canvas : any;
  public ctx;
  public chartColor;
  public chartEmail;
  public chartHours;
  public ChartHoriz;
  public mixedchart;
  public bubblechart;
  public lineChartData:Array<any>;
  public lineChartLabels:Array<any>;
  public lineChartType;
  infos: Info[];
  users: User[];
  tasks: Task[];
  data = new Array();
  count_by_categorie = new Array();
  totalrating_by_categorie = new Array();
  totalliking_by_categorie = new Array();
  rated_by_categorie = new Array();
  count_by_subcategorie_accessories = new Array();
  totalrating_by_subcategorie_accessories = new Array();
  totalliking_by_subcategorie_accessories = new Array();
  rated_by_subcategorie_accessories = new Array();
  count_by_subcategorie_apparel = new Array();
  totalrating_by_subcategorie_apparel = new Array();
  totalliking_by_subcategorie_apparel = new Array();
  rated_by_subcategorie_apparel = new Array();
  count_by_subcategorie_books = new Array();
  totalrating_by_subcategorie_books = new Array();
  totalliking_by_subcategorie_books = new Array();
  rated_by_subcategorie_books = new Array();
  count_by_subcategorie_footwear = new Array();
  totalrating_by_subcategorie_footwear = new Array();
  totalliking_by_subcategorie_footwear = new Array();
  rated_by_subcategorie_footwear = new Array();
  count_by_subcategorie_personal_care = new Array();
  totalrating_by_subcategorie_personal_care = new Array();
  totalliking_by_subcategorie_personal_care = new Array();
  rated_by_subcategorie_personal_care = new Array();
  count_by_subcategorie_phones = new Array();
  totalrating_by_subcategorie_phones = new Array();
  totalliking_by_subcategorie_phones = new Array();
  rated_by_subcategorie_phones = new Array();
  count_by_subcategorie_sporting = new Array();
  totalrating_by_subcategorie_sporting = new Array();
  totalliking_by_subcategorie_sporting = new Array();
  rated_by_subcategorie_sporting = new Array();
  UserId = JSON.parse(localStorage.getItem('user')).userId;
  subcategorie = new Array();

  constructor(private projetService: ProjetService, private http: HttpClient) { }

  ngOnInit(){
    console.log(JSON.parse(localStorage.getItem('user')).userId)
  

    this.projetService
    .popularproductsuser(this.UserId)
    .subscribe((data: Task[]) => {
      this.tasks = data;
    console.log(this.tasks[2].like)
    this.canvas = document.getElementById("mixedchart");
    this.ctx = this.canvas.getContext("2d");
    this.mixedchart = new Chart(this.ctx, {
      type: 'bar',
      data: {
        labels: ["Time Rated", "Rating","Liked"],
        datasets: [
          
          {
            label: this.tasks[0].Product_name,
            type: "bar",
            backgroundColor: "#000000",
            data: [this.tasks[0].Count,this.tasks[0].Mean,this.tasks[0].like],
          },
          {
            label: this.tasks[1].Product_name,
            type: "bar",
            backgroundColor: "#808080",
            data: [this.tasks[1].Count,this.tasks[1].Mean,this.tasks[1].like],
          },
          {
            label: this.tasks[2].Product_name,
            type: "bar",
            backgroundColor: "#FF9999",
            data: [this.tasks[2].Count,this.tasks[2].Mean,this.tasks[2].like],
          },
          {
            label: this.tasks[3].Product_name,
            type: "bar",
            backgroundColor: "#FF3333",
            data: [this.tasks[3].Count,this.tasks[3].Mean,this.tasks[3].like],
          },
          {
            label: this.tasks[4].Product_name,
            type: "bar",
            backgroundColor: "#CC0000",
            backgroundColorHover: "#3e95cd",
            data: [this.tasks[4].Count,this.tasks[4].Mean,this.tasks[4].like],
          }
        ]
      },
      options: {
        title: {
          display: false,
          text: 'Population growth (millions): Europe & Africa'
        },
        legend: { display: false }
      }
    });

    this.canvas = document.getElementById("ChartHoriz");
    this.ctx = this.canvas.getContext("2d");
    this.ChartHoriz = new Chart(this.ctx, {
    type: 'horizontalBar',
    data: { /*moxkila makayjibxi kolxi mafhamtxi elax*/
      labels: [this.tasks[0].Product_name,this.tasks[1].Product_name,this.tasks[2].Product_name,this.tasks[3].Product_name,this.tasks[4].Product_name],
      datasets: [
        {
          label: "",
          backgroundColor: ["#3e95cd", "#8e5ea2","#3cba9f","#e8c3b9","#c45850"],
          data: [this.tasks[0].like,this.tasks[1].like,this.tasks[2].like,this.tasks[3].like,this.tasks[4].like]
        }
      ]
    },
    options: {
      legend: { display: false },
      title: {
        display: false,
        text: 'Predicted world population (millions) in 2050'
      }
    }
});

    });


    this.projetService
    .infouser(this.UserId)
    .subscribe((data: Info[]) => {
      this.infos = data;
      this.infos.forEach(n => {
          this.count_by_categorie.push(n.accessoriescount);
          this.count_by_categorie.push(n.apparelcount);
          this.count_by_categorie.push(n.bookscount);
          this.count_by_categorie.push(n.footwearcount);
          this.count_by_categorie.push(n.personal_Carecount);
          this.count_by_categorie.push(n.phonescount);
          this.count_by_categorie.push(n.sportcount);

          this.totalrating_by_categorie.push(n.accessoriestotalrating);
          this.totalrating_by_categorie.push(n.appareltotalrating);
          this.totalrating_by_categorie.push(n.bookstotalrating);
          this.totalrating_by_categorie.push(n.footweartotalrating);
          this.totalrating_by_categorie.push(n.personal_Caretotalrating);
          this.totalrating_by_categorie.push(n.phonestotalrating);
          this.totalrating_by_categorie.push(n.sporttotalrating);

          this.totalliking_by_categorie.push(n.accessoriestotalliking);
          this.totalliking_by_categorie.push(n.appareltotalliking);
          this.totalliking_by_categorie.push(n.bookstotalliking);
          this.totalliking_by_categorie.push(n.footweartotalliking);
          this.totalliking_by_categorie.push(n.personal_Caretotalliking);
          this.totalliking_by_categorie.push(n.phonestotalliking);
          this.totalliking_by_categorie.push(n.sporttotalliking);

          this.rated_by_categorie.push(n.accessoriesrated);
          this.rated_by_categorie.push(n.apparelrated);
          this.rated_by_categorie.push(n.booksrated);
          this.rated_by_categorie.push(n.footwearrated);
          this.rated_by_categorie.push(n.personal_Carerated);
          this.rated_by_categorie.push(n.phonesrated);
          this.rated_by_categorie.push(n.sportrated);

          /* hna nzid arrays akhrin*/

          this.count_by_subcategorie_accessories.push(n.bags_Accessoriescount);
          this.count_by_subcategorie_accessories.push(n.beltscount);
          this.count_by_subcategorie_accessories.push(n.cufflinkscount);
          this.count_by_subcategorie_accessories.push(n.eyewear_Accessoriescount);
          this.count_by_subcategorie_accessories.push(n.giftcount);
          this.count_by_subcategorie_accessories.push(n.glovescount);
          this.count_by_subcategorie_accessories.push(n.headwear_Accessoriescount);
          this.count_by_subcategorie_accessories.push(n.jewellerycount);
          this.count_by_subcategorie_accessories.push(n.walletscount);
          this.count_by_subcategorie_accessories.push(n.watches_Accessoriescount);

          this.totalrating_by_subcategorie_accessories.push(n.bags_Accessoriestotalrating);
          this.totalrating_by_subcategorie_accessories.push(n.beltstotalrating);
          this.totalrating_by_subcategorie_accessories.push(n.cufflinkstotalrating);
          this.totalrating_by_subcategorie_accessories.push(n.eyewear_Accessoriestotalrating);
          this.totalrating_by_subcategorie_accessories.push(n.gifttotalrating);
          this.totalrating_by_subcategorie_accessories.push(n.glovestotalrating);
          this.totalrating_by_subcategorie_accessories.push(n.headwear_Accessoriestotalrating);
          this.totalrating_by_subcategorie_accessories.push(n.jewellerytotalrating);
          this.totalrating_by_subcategorie_accessories.push(n.walletstotalrating);
          this.totalrating_by_subcategorie_accessories.push(n.watches_Accessoriestotalrating);

          this.totalliking_by_subcategorie_accessories.push(n.bags_Accessoriestotalliking);
          this.totalliking_by_subcategorie_accessories.push(n.beltstotalliking);
          this.totalliking_by_subcategorie_accessories.push(n.cufflinkstotalliking);
          this.totalliking_by_subcategorie_accessories.push(n.eyewear_Accessoriestotalliking);
          this.totalliking_by_subcategorie_accessories.push(n.giftstotalliking);
          this.totalliking_by_subcategorie_accessories.push(n.glovestotalliking);
          this.totalliking_by_subcategorie_accessories.push(n.headwear_Accessoriestotalliking);
          this.totalliking_by_subcategorie_accessories.push(n.jewellerytotalliking);
          this.totalliking_by_subcategorie_accessories.push(n.walletstotalliking);
          this.totalliking_by_subcategorie_accessories.push(n.watches_Accessoriestotalliking);

          this.rated_by_subcategorie_accessories.push(n.bags_Accessoriesrated);
          this.rated_by_subcategorie_accessories.push(n.beltsrated);
          this.rated_by_subcategorie_accessories.push(n.cufflinksrated);
          this.rated_by_subcategorie_accessories.push(n.eyewear_Accessoriesrated);
          this.rated_by_subcategorie_accessories.push(n.giftsrated);
          this.rated_by_subcategorie_accessories.push(n.glovesrated);
          this.rated_by_subcategorie_accessories.push(n.headwear_Accessoriesrated);
          this.rated_by_subcategorie_accessories.push(n.jewelleryrated);
          this.rated_by_subcategorie_accessories.push(n.walletsrated);
          this.rated_by_subcategorie_accessories.push(n.watches_Accessoriesrated);

          this.count_by_subcategorie_apparel.push(n.apparel_setcount);
          this.count_by_subcategorie_apparel.push(n.bottomwearcount);
          this.count_by_subcategorie_apparel.push(n.dresscount);
          this.count_by_subcategorie_apparel.push(n.innerwearcount);
          this.count_by_subcategorie_apparel.push(n.loungewear_nightwearcount);
          this.count_by_subcategorie_apparel.push(n.sareecount);
          this.count_by_subcategorie_apparel.push(n.scarvescount);
          this.count_by_subcategorie_apparel.push(n.stolescount);
          this.count_by_subcategorie_apparel.push(n.tiescount);
          this.count_by_subcategorie_apparel.push(n.topwearcount);

          this.totalrating_by_subcategorie_apparel.push(n.innerweartotalrating);
          this.totalrating_by_subcategorie_apparel.push(n.scarvestotalrating);
          this.totalrating_by_subcategorie_apparel.push(n.bottomweatotalrrating);
          this.totalrating_by_subcategorie_apparel.push(n.topweartotalrating);
          this.totalrating_by_subcategorie_apparel.push(n.apparel_settotalrating);
          this.totalrating_by_subcategorie_apparel.push(n.dresstotalrating);
          this.totalrating_by_subcategorie_apparel.push(n.tiestotalrating);
          this.totalrating_by_subcategorie_apparel.push(n.stolestotalrating);
          this.totalrating_by_subcategorie_apparel.push(n.loungewear_nightweartotalrating);
          this.totalrating_by_subcategorie_apparel.push(n.sareetotalrating);

          this.totalliking_by_subcategorie_apparel.push(n.innerweartotalliking);
          this.totalliking_by_subcategorie_apparel.push(n.scarvestotalliking);
          this.totalliking_by_subcategorie_apparel.push(n.bottomweartotalliking);
          this.totalliking_by_subcategorie_apparel.push(n.topweartotalliking);
          this.totalliking_by_subcategorie_apparel.push(n.apparel_settotalliking);
          this.totalliking_by_subcategorie_apparel.push(n.dresstotalliking);
          this.totalliking_by_subcategorie_apparel.push(n.tiestotalliking);
          this.totalliking_by_subcategorie_apparel.push(n.stolestotalliking);
          this.totalliking_by_subcategorie_apparel.push(n.loungewear_nightweartotalliking);
          this.totalliking_by_subcategorie_apparel.push(n.sareetotalliking);

          this.rated_by_subcategorie_apparel.push(n.innerwearrated);
          this.rated_by_subcategorie_apparel.push(n.scarvesrated);
          this.rated_by_subcategorie_apparel.push(n.bottomwearrated);
          this.rated_by_subcategorie_apparel.push(n.topwearrated);
          this.rated_by_subcategorie_apparel.push(n.apparel_setrated);
          this.rated_by_subcategorie_apparel.push(n.dressrated);
          this.rated_by_subcategorie_apparel.push(n.tiesrated);
          this.rated_by_subcategorie_apparel.push(n.stolesrated);
          this.rated_by_subcategorie_apparel.push(n.loungewear_nightwearrated);
          this.rated_by_subcategorie_apparel.push(n.sareerated);

          this.count_by_subcategorie_books.push(n.generalcount);
          this.count_by_subcategorie_books.push(n.childrencount);
          this.count_by_subcategorie_books.push(n.classicscount);
          this.count_by_subcategorie_books.push(n.crimecount);
          this.count_by_subcategorie_books.push(n.families_relationshipcount);
          this.count_by_subcategorie_books.push(n.fantasycount);
          this.count_by_subcategorie_books.push(n.literaturecount);
          this.count_by_subcategorie_books.push(n.otherscount);
          this.count_by_subcategorie_books.push(n.romancecount);
          this.count_by_subcategorie_books.push(n.science_fictioncount);
          this.count_by_subcategorie_books.push(n.short_storiescount);
          this.count_by_subcategorie_books.push(n.young_adultscount);

          this.totalrating_by_subcategorie_books.push(n.generaltotalrating);
          this.totalrating_by_subcategorie_books.push(n.childrentotalrating);
          this.totalrating_by_subcategorie_books.push(n.classicstotalrating);
          this.totalrating_by_subcategorie_books.push(n.crimetotalrating);
          this.totalrating_by_subcategorie_books.push(n.families_relationshiptotalrating);
          this.totalrating_by_subcategorie_books.push(n.fantasytotalrating);
          this.totalrating_by_subcategorie_books.push(n.literaturetotalrating);
          this.totalrating_by_subcategorie_books.push(n.otherstotalrating);
          this.totalrating_by_subcategorie_books.push(n.romancetotalrating);
          this.totalrating_by_subcategorie_books.push(n.science_fictiontotalrating);
          this.totalrating_by_subcategorie_books.push(n.short_storiestotalrating);
          this.totalrating_by_subcategorie_books.push(n.young_adultstotalrating);

          this.totalliking_by_subcategorie_books.push(n.generaltotalliking);
          this.totalliking_by_subcategorie_books.push(n.childrentotalliking);
          this.totalliking_by_subcategorie_books.push(n.classicstotalliking);
          this.totalliking_by_subcategorie_books.push(n.crimetotalliking);
          this.totalliking_by_subcategorie_books.push(n.families_relationshiptotalliking);
          this.totalliking_by_subcategorie_books.push(n.fantasytotalliking);
          this.totalliking_by_subcategorie_books.push(n.literaturetotalliking);
          this.totalliking_by_subcategorie_books.push(n.otherstotalliking);
          this.totalliking_by_subcategorie_books.push(n.romancetotalliking);
          this.totalliking_by_subcategorie_books.push(n.science_fictiontotalliking);
          this.totalliking_by_subcategorie_books.push(n.short_storiestotalliking);
          this.totalliking_by_subcategorie_books.push(n.young_adultstotalliking);

          this.rated_by_subcategorie_books.push(n.generalrated);
          this.rated_by_subcategorie_books.push(n.childrenrated);
          this.rated_by_subcategorie_books.push(n.classicsrated);
          this.rated_by_subcategorie_books.push(n.crimerated);
          this.rated_by_subcategorie_books.push(n.families_relationshiprated);
          this.rated_by_subcategorie_books.push(n.fantasyrated);
          this.rated_by_subcategorie_books.push(n.literaturerated);
          this.rated_by_subcategorie_books.push(n.othersrated);
          this.rated_by_subcategorie_books.push(n.romancerated);
          this.rated_by_subcategorie_books.push(n.science_fictionrated);
          this.rated_by_subcategorie_books.push(n.short_storiesrated);
          this.rated_by_subcategorie_books.push(n.young_adultsrated);

          this.count_by_subcategorie_footwear.push(n.flip_flopscount);
          this.count_by_subcategorie_footwear.push(n.sandalcount);
          this.count_by_subcategorie_footwear.push(n.shoe_accessoriescount);
          this.count_by_subcategorie_footwear.push(n.shoescount);
          this.count_by_subcategorie_footwear.push(n.sockscount);

          this.totalrating_by_subcategorie_footwear.push(n.flip_flopstotalrating);
          this.totalrating_by_subcategorie_footwear.push(n.sandaltotalrating);
          this.totalrating_by_subcategorie_footwear.push(n.shoe_accessoriestotalrating);
          this.totalrating_by_subcategorie_footwear.push(n.shoestotalrating);
          this.totalrating_by_subcategorie_footwear.push(n.sockstotalrating);

          this.totalliking_by_subcategorie_footwear.push(n.flip_flopstotalliking);
          this.totalliking_by_subcategorie_footwear.push(n.sandaltotalliking);
          this.totalliking_by_subcategorie_footwear.push(n.shoe_accessoriestotalliking);
          this.totalliking_by_subcategorie_footwear.push(n.shoestotalliking);
          this.totalliking_by_subcategorie_footwear.push(n.sockstotalliking);

          this.rated_by_subcategorie_footwear.push(n.flip_flopsrated);
          this.rated_by_subcategorie_footwear.push(n.sandalrated);
          this.rated_by_subcategorie_footwear.push(n.shoe_accessoriesrated);
          this.rated_by_subcategorie_footwear.push(n.shoesrated);
          this.rated_by_subcategorie_footwear.push(n.socksrated);

          this.count_by_subcategorie_personal_care.push(n.bath_bodycount);
          this.count_by_subcategorie_personal_care.push(n.eyescount);
          this.count_by_subcategorie_personal_care.push(n.fragrancecount);
          this.count_by_subcategorie_personal_care.push(n.haircount);
          this.count_by_subcategorie_personal_care.push(n.lipscount);
          this.count_by_subcategorie_personal_care.push(n.makeupcount);
          this.count_by_subcategorie_personal_care.push(n.nailscount);
          this.count_by_subcategorie_personal_care.push(n.skin_carecount);

          this.totalrating_by_subcategorie_personal_care.push(n.bath_bodytotalrating);
          this.totalrating_by_subcategorie_personal_care.push(n.eyestotalrating);
          this.totalrating_by_subcategorie_personal_care.push(n.fragrancetotalrating);
          this.totalrating_by_subcategorie_personal_care.push(n.hairtotalrating);
          this.totalrating_by_subcategorie_personal_care.push(n.lipstotalrating);
          this.totalrating_by_subcategorie_personal_care.push(n.makeuptotalrating);
          this.totalrating_by_subcategorie_personal_care.push(n.nailstotalrating);
          this.totalrating_by_subcategorie_personal_care.push(n.skin_caretotalrating);

          this.totalliking_by_subcategorie_personal_care.push(n.bath_bodytotalliking);
          this.totalliking_by_subcategorie_personal_care.push(n.eyestotalliking);
          this.totalliking_by_subcategorie_personal_care.push(n.fragrancetotalliking);
          this.totalliking_by_subcategorie_personal_care.push(n.hairtotalliking);
          this.totalliking_by_subcategorie_personal_care.push(n.lipstotalliking);
          this.totalliking_by_subcategorie_personal_care.push(n.makeuptotalliking);
          this.totalliking_by_subcategorie_personal_care.push(n.nailstotalliking);
          this.totalliking_by_subcategorie_personal_care.push(n.skin_caretotalliking);

          this.rated_by_subcategorie_personal_care.push(n.bath_bodyrated);
          this.rated_by_subcategorie_personal_care.push(n.eyesrated);
          this.rated_by_subcategorie_personal_care.push(n.fragrancerated);
          this.rated_by_subcategorie_personal_care.push(n.hairrated);
          this.rated_by_subcategorie_personal_care.push(n.lipsrated);
          this.rated_by_subcategorie_personal_care.push(n.makeuprated);
          this.rated_by_subcategorie_personal_care.push(n.nailsrated);
          this.rated_by_subcategorie_personal_care.push(n.skin_carerated);
          
          this.count_by_subcategorie_phones.push(n.androidcount);
          this.count_by_subcategorie_phones.push(n.applecount);
          this.count_by_subcategorie_phones.push(n.honorcount);
          this.count_by_subcategorie_phones.push(n.huaweicount);
          this.count_by_subcategorie_phones.push(n.nokiacount);
          this.count_by_subcategorie_phones.push(n.onepluscount);
          this.count_by_subcategorie_phones.push(n.oppocount);
          this.count_by_subcategorie_phones.push(n.redmicount);
          this.count_by_subcategorie_phones.push(n.samsungcount);
          this.count_by_subcategorie_phones.push(n.sonycount);
          this.count_by_subcategorie_phones.push(n.vivocount);

          this.totalrating_by_subcategorie_phones.push(n.androidtotalrating);
          this.totalrating_by_subcategorie_phones.push(n.appletotalrating);
          this.totalrating_by_subcategorie_phones.push(n.honortotalrating);
          this.totalrating_by_subcategorie_phones.push(n.huaweitotalrating);
          this.totalrating_by_subcategorie_phones.push(n.nokiatotalrating);
          this.totalrating_by_subcategorie_phones.push(n.oneplustotalrating);
          this.totalrating_by_subcategorie_phones.push(n.oppototalrating);
          this.totalrating_by_subcategorie_phones.push(n.redmitotalrating);
          this.totalrating_by_subcategorie_phones.push(n.samsungtotalrating);
          this.totalrating_by_subcategorie_phones.push(n.sonytotalrating);
          this.totalrating_by_subcategorie_phones.push(n.vivototalrating);

          this.totalliking_by_subcategorie_phones.push(n.androidtotalliking);
          this.totalliking_by_subcategorie_phones.push(n.appletotalliking);
          this.totalliking_by_subcategorie_phones.push(n.honortotalliking);
          this.totalliking_by_subcategorie_phones.push(n.huaweitotalliking);
          this.totalliking_by_subcategorie_phones.push(n.nokiatotalliking);
          this.totalliking_by_subcategorie_phones.push(n.oneplustotalliking);
          this.totalliking_by_subcategorie_phones.push(n.oppototalliking);
          this.totalliking_by_subcategorie_phones.push(n.redmitotalliking);
          this.totalliking_by_subcategorie_phones.push(n.samsungtotalliking);
          this.totalliking_by_subcategorie_phones.push(n.sonytotalliking);
          this.totalliking_by_subcategorie_phones.push(n.vivototalliking);

          this.rated_by_subcategorie_phones.push(n.androidrated);
          this.rated_by_subcategorie_phones.push(n.applerated);
          this.rated_by_subcategorie_phones.push(n.honorrated);
          this.rated_by_subcategorie_phones.push(n.huaweirated)
          this.rated_by_subcategorie_phones.push(n.nokiarated);
          this.rated_by_subcategorie_phones.push(n.oneplusrated);
          this.rated_by_subcategorie_phones.push(n.opporated);
          this.rated_by_subcategorie_phones.push(n.redmirated);
          this.rated_by_subcategorie_phones.push(n.samsungrated);
          this.rated_by_subcategorie_phones.push(n.sonyrated);
          this.rated_by_subcategorie_phones.push(n.vivorated);
    
          this.count_by_subcategorie_sporting.push(n.bags_Sportingcount);
          this.count_by_subcategorie_sporting.push(n.eyewear_Sportingcount);
          this.count_by_subcategorie_sporting.push(n.headwear_Sportingcount);
          this.count_by_subcategorie_sporting.push(n.sports_equipmentcount);
          this.count_by_subcategorie_sporting.push(n.watches_Sportingcount);
          this.count_by_subcategorie_sporting.push(n.water_bottlecount);

          this.totalrating_by_subcategorie_sporting.push(n.bags_Sportingtotalrating);
          this.totalrating_by_subcategorie_sporting.push(n.eyewear_Sportingtotalrating);
          this.totalrating_by_subcategorie_sporting.push(n.headwear_Sportingtotalrating);
          this.totalrating_by_subcategorie_sporting.push(n.sports_equipmenttotalrating);
          this.totalrating_by_subcategorie_sporting.push(n.watches_Sportingtotalrating);
          this.totalrating_by_subcategorie_sporting.push(n.water_bottletotalrating);

          this.totalliking_by_subcategorie_sporting.push(n.bags_Sportingtotalliking);
          this.totalliking_by_subcategorie_sporting.push(n.eyewear_Sportingtotalliking);
          this.totalliking_by_subcategorie_sporting.push(n.headwear_Sportingtotalliking);
          this.totalliking_by_subcategorie_sporting.push(n.sports_equipmenttotalliking);
          this.totalliking_by_subcategorie_sporting.push(n.watches_Sportingtotalliking);
          this.totalliking_by_subcategorie_sporting.push(n.water_bottletotalliking);

          this.rated_by_subcategorie_sporting.push(n.bags_Sportingrated);
          this.rated_by_subcategorie_sporting.push(n.eyewear_Sportingrated);
          this.rated_by_subcategorie_sporting.push(n.headwear_Sportingrated);
          this.rated_by_subcategorie_sporting.push(n.sports_equipmentrated);
          this.rated_by_subcategorie_sporting.push(n.watches_Sportingrated);
          this.rated_by_subcategorie_sporting.push(n.water_bottlerated);

          
          
        
        
      });
      
      this.count_by_categorie = this.count_by_categorie.filter(f => f !== undefined)
      this.totalrating_by_categorie = this.totalrating_by_categorie.filter(f => f !== undefined)
      this.totalliking_by_categorie = this.totalliking_by_categorie.filter(f => f !== undefined)
      this.rated_by_categorie = this.rated_by_categorie.filter(f => f !== undefined)
      this.count_by_subcategorie_accessories = this.count_by_subcategorie_accessories.filter(f => f !== undefined)
      this.totalrating_by_subcategorie_accessories = this.totalrating_by_subcategorie_accessories.filter(f => f !== undefined)
      this.totalliking_by_subcategorie_accessories = this.totalliking_by_subcategorie_accessories.filter(f => f !== undefined)
      this.rated_by_subcategorie_accessories = this.rated_by_subcategorie_accessories.filter(f => f !== undefined)
      this.count_by_subcategorie_apparel = this.count_by_subcategorie_apparel.filter(f => f !== undefined)
      this.totalrating_by_subcategorie_apparel = this.totalrating_by_subcategorie_apparel.filter(f => f !== undefined)
      this.totalliking_by_subcategorie_apparel = this.totalliking_by_subcategorie_apparel.filter(f => f !== undefined)
      this.rated_by_subcategorie_apparel = this.rated_by_subcategorie_apparel.filter(f => f !== undefined)
      this.count_by_subcategorie_books = this.count_by_subcategorie_books.filter(f => f !== undefined)
      this.totalrating_by_subcategorie_books = this.totalrating_by_subcategorie_books.filter(f => f !== undefined)
      this.totalliking_by_subcategorie_books = this.totalliking_by_subcategorie_books.filter(f => f !== undefined)
      this.rated_by_subcategorie_books = this.rated_by_subcategorie_books.filter(f => f !== undefined)
      this.count_by_subcategorie_footwear = this.count_by_subcategorie_footwear.filter(f => f !== undefined)
      this.totalrating_by_subcategorie_footwear = this.totalrating_by_subcategorie_footwear.filter(f => f !== undefined)
      this.totalliking_by_subcategorie_footwear = this.totalliking_by_subcategorie_footwear.filter(f => f !== undefined)
      this.rated_by_subcategorie_footwear = this.rated_by_subcategorie_footwear.filter(f => f !== undefined)
      this.count_by_subcategorie_personal_care = this.count_by_subcategorie_personal_care.filter(f => f !== undefined)
      this.totalrating_by_subcategorie_personal_care = this.totalrating_by_subcategorie_personal_care.filter(f => f !== undefined)
      this.totalliking_by_subcategorie_personal_care = this.totalliking_by_subcategorie_personal_care.filter(f => f !== undefined)
      this.rated_by_subcategorie_personal_care = this.rated_by_subcategorie_personal_care.filter(f => f !== undefined)
      this.count_by_subcategorie_phones = this.count_by_subcategorie_phones.filter(f => f !== undefined)
      this.totalrating_by_subcategorie_phones = this.totalrating_by_subcategorie_phones.filter(f => f !== undefined)
      this.totalliking_by_subcategorie_phones = this.totalliking_by_subcategorie_phones.filter(f => f !== undefined)
      this.rated_by_subcategorie_phones = this.rated_by_subcategorie_phones.filter(f => f !== undefined)
      this.count_by_subcategorie_sporting = this.count_by_subcategorie_sporting.filter(f => f !== undefined)
      this.totalrating_by_subcategorie_sporting = this.totalrating_by_subcategorie_sporting.filter(f => f !== undefined)
      this.totalliking_by_subcategorie_sporting = this.totalliking_by_subcategorie_sporting.filter(f => f !== undefined)
      this.rated_by_subcategorie_sporting = this.rated_by_subcategorie_sporting.filter(f => f !== undefined)
      
      
      
  
      if(this.count_by_subcategorie_accessories[0] != 0) {this.subcategorie.push('bags(Accessories)')}
      if(this.count_by_subcategorie_accessories[1] != 0) {this.subcategorie.push('belts')}
      if(this.count_by_subcategorie_accessories[2]  != 0) {this.subcategorie.push('cufflinks')}
      if(this.count_by_subcategorie_accessories[3]  != 0) {this.subcategorie.push('eyewear(Accessories)')}
      if(this.count_by_subcategorie_accessories[4]  != 0) {this.subcategorie.push('gift set')}
      if(this.count_by_subcategorie_accessories[5]  != 0) {this.subcategorie.push('gloves')}
      if(this.count_by_subcategorie_accessories[6]  != 0) {this.subcategorie.push('headwear(Accessories)')} 
      if(this.count_by_subcategorie_accessories[7]  != 0) {this.subcategorie.push('jewellery')}
      if(this.count_by_subcategorie_accessories[8]  != 0) {this.subcategorie.push('wallets')}
      if(this.count_by_subcategorie_accessories[9]  != 0) {this.subcategorie.push('watches(Accessories)')}
      if(this.count_by_subcategorie_apparel[0]  != 0) {this.subcategorie.push('apparel set')}
      if(this.count_by_subcategorie_apparel[1]  != 0) {this.subcategorie.push('bottomwear')}
      if(this.count_by_subcategorie_apparel[2]  != 0) {this.subcategorie.push('dress')}
      if(this.count_by_subcategorie_apparel[3] != 0) {this.subcategorie.push('innerwear')}
      if(this.count_by_subcategorie_apparel[4]  != 0) {this.subcategorie.push('loungewear and nightwear')}
      if(this.count_by_subcategorie_apparel[5]  != 0) {this.subcategorie.push('saree')} 
      if(this.count_by_subcategorie_apparel[6] != 0) {this.subcategorie.push('scarves')}
      if(this.count_by_subcategorie_apparel[7]  != 0) {this.subcategorie.push('stoles')}
      if(this.count_by_subcategorie_apparel[8]  != 0) {this.subcategorie.push('ties')} 
      if(this.count_by_subcategorie_apparel[9]  != 0) {this.subcategorie.push('topwear')}
      
     
      
      
      
      
      if(this.count_by_subcategorie_books[0] != 0) {this.subcategorie.push('general')}
      if(this.count_by_subcategorie_books[1] != 0) {this.subcategorie.push('children')}
      if(this.count_by_subcategorie_books[2]  != 0) {this.subcategorie.push('classics')}
      if(this.count_by_subcategorie_books[3]  != 0) {this.subcategorie.push('crime')}
      if(this.count_by_subcategorie_books[4]  != 0) {this.subcategorie.push('families relationship')}
      if(this.count_by_subcategorie_books[5]  != 0) {this.subcategorie.push('fantasy')}
      if(this.count_by_subcategorie_books[6]  != 0) {this.subcategorie.push('literature')} 
      if(this.count_by_subcategorie_books[7]  != 0) {this.subcategorie.push('others')}
      if(this.count_by_subcategorie_books[8]  != 0) {this.subcategorie.push('romance')}
      if(this.count_by_subcategorie_books[9]  != 0) {this.subcategorie.push('science fiction')}
      if(this.count_by_subcategorie_books[10]  != 0) {this.subcategorie.push('short stories')}
      if(this.count_by_subcategorie_books[11]  != 0) {this.subcategorie.push('young adults')} 
      if(this.count_by_subcategorie_footwear[0] != 0) {this.subcategorie.push('flip flops')}
      if(this.count_by_subcategorie_footwear[1] != 0) {this.subcategorie.push('sandal')}
      if(this.count_by_subcategorie_footwear[2]  != 0) {this.subcategorie.push('shoe accessories')}
      if(this.count_by_subcategorie_footwear[3]  != 0) {this.subcategorie.push('shoes')}
      if(this.count_by_subcategorie_footwear[4]  != 0) {this.subcategorie.push('socks')}
      if(this.count_by_subcategorie_personal_care[0] != 0) {this.subcategorie.push('bath body')}
      if(this.count_by_subcategorie_personal_care[1] != 0) {this.subcategorie.push('eyes')}
      if(this.count_by_subcategorie_personal_care[2]  != 0) {this.subcategorie.push('fragrance')}
      if(this.count_by_subcategorie_personal_care[3]  != 0) {this.subcategorie.push('hair')}
      if(this.count_by_subcategorie_personal_care[4]  != 0) {this.subcategorie.push('lips')}
      if(this.count_by_subcategorie_personal_care[5]  != 0) {this.subcategorie.push('makeup')}
      if(this.count_by_subcategorie_personal_care[6]  != 0) {this.subcategorie.push('nails')} 
      if(this.count_by_subcategorie_personal_care[7]  != 0) {this.subcategorie.push('skin care')}
      if(this.count_by_subcategorie_phones[0] != 0) {this.subcategorie.push('android')}
      if(this.count_by_subcategorie_phones[1] != 0) {this.subcategorie.push('apple')}
      if(this.count_by_subcategorie_phones[2]  != 0) {this.subcategorie.push('honor')}
      if(this.count_by_subcategorie_phones[3]  != 0) {this.subcategorie.push('huawei')}
      if(this.count_by_subcategorie_phones[4]  != 0) {this.subcategorie.push('nokia')}
      if(this.count_by_subcategorie_phones[5]  != 0) {this.subcategorie.push('oneplus')}
      if(this.count_by_subcategorie_phones[6]  != 0) {this.subcategorie.push('oppo')} 
      if(this.count_by_subcategorie_phones[7]  != 0) {this.subcategorie.push('redmi')}
      if(this.count_by_subcategorie_phones[8]  != 0) {this.subcategorie.push('samsung')}
      if(this.count_by_subcategorie_phones[9]  != 0) {this.subcategorie.push('sony')}
      if(this.count_by_subcategorie_phones[10]  != 0) {this.subcategorie.push('vivo')}
      if(this.count_by_subcategorie_sporting[0] != 0) {this.subcategorie.push('bags(Sporting)')}
      if(this.count_by_subcategorie_sporting[1] != 0) {this.subcategorie.push('eyewear(Sporting)')}
      if(this.count_by_subcategorie_sporting[2]  != 0) {this.subcategorie.push('headwear(Sporting)')}
      if(this.count_by_subcategorie_sporting[3]  != 0) {this.subcategorie.push('Sport Equipement')}
      if(this.count_by_subcategorie_sporting[4]  != 0) {this.subcategorie.push('watches(Sporting)')}
      if(this.count_by_subcategorie_sporting[5]  != 0) {this.subcategorie.push('water bottle')}


      var speedCanvas = document.getElementById("speedChart");

      var dataFirst = {
        data: this.totalliking_by_categorie,
        fill: false,
        borderColor: '#E125D4',
        backgroundColor: 'transparent',
        pointBorderColor: '#E125D4',
        pointRadius: 4,
        pointHoverRadius: 4,
        pointBorderWidth: 8,
      };
      

      var dataSecond = {
        data: this.rated_by_categorie,
        fill: false,
        borderColor: '#ff0000',
        backgroundColor: 'transparent',
        pointBorderColor: '#ff0000',
        pointRadius: 4,
        pointHoverRadius: 4,
        pointBorderWidth: 8
      };

      var dataThird = {
        data: this.totalrating_by_categorie,
        fill: false,
        borderColor: '#00ff00',
        backgroundColor: 'transparent',
        pointBorderColor: '#00ff00',
        pointRadius: 4,
        pointHoverRadius: 4,
        pointBorderWidth: 8
      };

      var dataFourth = {
        data: this.count_by_categorie,
        fill: false,
        borderColor: '#51CACF',
        backgroundColor: 'transparent',
        pointBorderColor: '#c45850',
        pointRadius: 4,
        pointHoverRadius: 4,
        pointBorderWidth: 8
      };

      var speedData = {
        labels: ['Accessories', 'Apparel', 'Books', 'Footwear', 'Personal Care', 'Phones', 'Sporting Goods'],
        datasets: [dataFirst, dataSecond,dataThird,dataFourth]
      };

      var chartOptions = {
        legend: {
          display: false,
          position: 'top'
        }
      };

      var lineChart = new Chart(speedCanvas, {
        type: 'line',
        hover: false,
        data: speedData,
        options: chartOptions
      });
    

    this.canvas = document.getElementById("chartEmail");
    this.ctx = this.canvas.getContext("2d");
    this.chartEmail = new Chart(this.ctx, {
      type: 'bar',
      data: {
        labels: ["Bags Accessories", "Belts","Cufflinks","Eyewear Accessories","Gift","Gloves","Headwear Accessories","Jewellery","Wallets","Watches Accessories","Apparel Set","Bottomwear","Dress","Innerwear","loungewear & Nightwear","Saree","Scarves","Stoles","Ties","Topwear","Children","Classics","Crime","Families Relationship","Fantasy","General","Literature","Others Books","Romance","Science Fiction","Short Stories","Young Adults","Flip Flops","Sandal","Shoe Accessories","Shoes","Socks","Bath Body","Eyes","Fragrance","Hair","Lips","Makeup","Nails","Skin Care","Android","Apple","Honor","Huawei","Nokia","Oneplus","Oppo","Redmi","Samsung","Sony","Vivo","Bags Sporting","Eyewear Sporting","Headwear Sporting","Sports Equipment","Watches Sporting","Water bottle"],
        datasets: [
          {
            label: "Population (millions)",
            backgroundColor: ["#3e95cd", "#8e5ea2","#3cba9f","#e8c3b9","#c45850","#D2B48C", "#4682B4","#00FF7F","#3e95cd","#2E8B57","#9ACD32", "#FF0000","#800080","#B0E0E6","#CD853F","#7B68EE","#98FB98","#FF4500", "#808000","#FDF5E6","#000080","#FFE4B5","#7B68EE", "#FFE4B5","#FFE4B5","#DB7093","#EE82EE","#FF6347","#800000","#FFFFE0","#B0C4DE","#FFA07A","#F08080","#4B0082","#F0E68C","#FF69B4","#FFD700","#FF00FF","#228B22","#B22222","#9400D3","#8B0000","#FF8C00","#006400","#008B8B","#FFF8DC","#D2691E","#A52A2A","#0000FF","#000000","#7FFF00","#8B008B","#E9967A","#B22222","#4B0082","#7CFC00","#00FF00","#808000","#A0522D","#DA70D6","#8B0000"],
            data: this.count_by_subcategorie_accessories.concat(this.count_by_subcategorie_apparel,this.count_by_subcategorie_books,this.count_by_subcategorie_footwear,this.count_by_subcategorie_personal_care,this.count_by_subcategorie_phones,this.count_by_subcategorie_sporting)
          }
        ]
      },
      options: {
        legend: { display: false },
        title: {
          display: true,
          text: 'Count Posted Product By SubCategorie'
        }
      }
  });


  






  });
  this.canvas = document.getElementById("lineChartExample");
    this.ctx = this.canvas.getContext("2d");

    this.lineChartData = [
        {
          label: "Count",
          pointBorderWidth: 2,
          pointHoverRadius: 4,
          pointHoverBorderWidth: 1,
          pointRadius: 4,
          fill: true,
          borderWidth: 2,
          data: this.count_by_categorie
        }
      ];
    this.lineChartLabels = ['Accessories', 'Apparel', 'Books', 'Footwear', 'Personal Care', 'Phones', 'Sporting Goods'];
    this.lineChartType = 'line';
}

}
