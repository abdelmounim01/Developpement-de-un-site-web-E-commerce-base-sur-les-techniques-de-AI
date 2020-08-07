import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { PortfolioImagesvisiteurComponent } from './portfolio-images.component';

describe('PortfolioImagesComponent', () => {
  let component: PortfolioImagesvisiteurComponent;
  let fixture: ComponentFixture<PortfolioImagesvisiteurComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ PortfolioImagesvisiteurComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(PortfolioImagesvisiteurComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
